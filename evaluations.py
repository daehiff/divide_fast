import itertools
import os.path
import sys
import time
import warnings

import networkx as nx
import numpy as np
import pytket
import pyzx.simplify
import qiskit
from pauliopt.topologies import Topology
from pauliopt.utils import pi
from pytket._tket.circuit import PauliExpBox
from pytket._tket.passes import SequencePass, PauliSimp, PlacementPass, RoutingPass
from pytket._tket.pauli import Pauli
from pytket._tket.predicates import CompilationUnit, ConnectivityPredicate
from pytket._tket.routing import GraphPlacement
from pytket._tket.transform import Transform
from pytket.utils import circuit_to_symbolic_unitary
from pyzx import Mat2
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from sympy import Symbol, latex

from optimization.optimisation import optimize_zx_poly
from steiner import rec_steiner_gauss
from synth.divide_utils import find_minimal_cx_assignment, ZXPolyRegion
from synth.utils import get_matrix, get_phases

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pauliopt.phase import PhaseCircuit, OptimizedPhaseCircuit, Z, X

import pytket.passes

from ZXPhasePoly import ZXPhasePoly, RemainingCNOTTracker
from architecture import Architecture, create_architecture, FULLY_CONNECTED, LINE, CIRCLE, SQUARE
from synth.synth_divide import get_count, get_non_matching_cnots, architecture_to_topology, synth_divide_and_conquer, \
    repeat_circuit
from utils import qiskit_to_tket, qiskit_pyzx, pyzx_to_tket, tket_to_qiskit, \
    verify_equality


class MetaData:
    def __init__(self, n_reps, n_qubits, algorithm, arch_name):
        self.n_qubits = n_qubits
        self.algorithm = algorithm
        self.n_reps = n_reps
        self.arch_name = arch_name

    def __repr__(self):
        tmp = [f"{self.__getattribute__(a)}" for a in dir(self) if not a.startswith('__')]
        return "/".join(tmp)


class RandomMetaData(MetaData):
    def __init__(self, n_reps, n_qubits, n_pgs, algorithm, arch_name, max_legs):
        super().__init__(n_reps, n_qubits, algorithm, arch_name)
        self.n_pgs = n_pgs
        self.max_legs = max_legs


class MaxCutMetadata(MetaData):
    def __init__(self, n_reps, n_qubits, algorithm, arch_name, p_edge, n_circ_reps):
        super().__init__(n_reps, n_qubits, algorithm, arch_name)
        self.p_edge = p_edge
        self.n_circ_reps = n_circ_reps


def generate_random_zx_polynomial_limit_legs(n_qubits, n_pgs, arch_name, max_legs=4, min_legs=1):
    architecture = create_architecture(arch_name, n_qubits=n_qubits)

    phase_circuit = PhaseCircuit.random(n_qubits, n_pgs, max_legs=max_legs, min_legs=min_legs)
    qc = ZXPolyRegion.from_zx_polynomial(ZXPhasePoly.from_pauli_op(phase_circuit, architecture)).to_qiskit()
    return qc, phase_circuit, architecture


def generate_random_zx_polynomial(n_qubits, n_pgs, arch_name, max_legs=None, min_legs=1):
    if max_legs is None:
        max_legs = n_qubits
    else:
        max_legs = max(1, int(n_qubits * max_legs))

    architecture = create_architecture(arch_name, n_qubits=n_qubits)

    phase_circuit = PhaseCircuit.random(n_qubits, n_pgs, max_legs=max_legs, min_legs=min_legs)
    qc = ZXPolyRegion.from_zx_polynomial(ZXPhasePoly.from_pauli_op(phase_circuit, architecture)).to_qiskit()
    return qc, phase_circuit, architecture


def compute_reduction(count_in, count_out):
    if count_in == 0:
        return 0.0
    else:
        return (count_in - count_out) / count_in


def get_circuit_stats(circ_in, circ_out, arch: Architecture, md: MetaData, time_diff, scenario="random",
                      default_cx_count=None):
    if isinstance(circ_out, pytket.Circuit):
        circ_out = tket_to_qiskit(circ_out)
    if isinstance(circ_in, pytket.Circuit):
        circ_in = tket_to_qiskit(circ_in)
    if default_cx_count is None:
        default_cx_count = get_count(circ_out, type="two_qubit")
    global_data = {
        "two_qubit_reduction": compute_reduction(get_count(circ_in, type="two_qubit"),
                                                 default_cx_count),
        "one_qubit_reduction": compute_reduction(get_count(circ_in, type="one_qubit"),
                                                 get_count(circ_out, type="one_qubit")),
        "cx_count_out": get_count(circ_out, type="two_qubit"),
        "cx_count_in": get_count(circ_in, type="two_qubit"),
        "one_qubit_count_out": get_count(circ_out, type="one_qubit"),
        "one_qubit_count_in": get_count(circ_in, type="one_qubit"),
        "n_qubits": md.n_qubits,
        "n_rep": md.n_reps,
        "non_matching_cx": get_non_matching_cnots(circ_out, arch),
        "time_diff": time_diff,
        "arch": md.arch_name,
        "algorithm": md.algorithm
    }
    if not os.path.isdir(f"./evaluations/{scenario}/circuits/{md.__repr__()}"):
        os.makedirs(f"./evaluations/{scenario}/circuits/{md.__repr__()}")
    circ_in.qasm(filename=f"./evaluations/{scenario}/circuits/{md.__repr__()}/circ_in.qasm")
    circ_out.qasm(filename=f"./evaluations/{scenario}/circuits/{md.__repr__()}/circ_out.qasm")

    if isinstance(md, RandomMetaData):
        return global_data | {"n_pgs": md.n_pgs, "max_legs": md.max_legs}
    elif isinstance(md, MaxCutMetadata):
        return global_data | {"p_edge": md.p_edge, "circ_reps": md.n_circ_reps}


def simplify_pyzx(circuit: qiskit.QuantumCircuit):
    g = qiskit_pyzx(circuit).to_graph()
    pyzx.simplify.full_reduce(g)
    circuit = pyzx.extract_circuit(g.copy()).to_basic_gates()
    outcirc = pyzx_to_tket(circuit)
    Transform.RebaseToTket().apply(outcirc)
    Transform.RebaseToRzRx().apply(outcirc)
    return outcirc


def simplify_pyzx_route_tket(circuit: qiskit.QuantumCircuit, arch: Architecture):
    g = qiskit_pyzx(circuit).to_graph()
    pyzx.simplify.full_reduce(g)
    circuit = pyzx.extract_circuit(g.copy()).to_basic_gates()
    circuit = pyzx_to_tket(circuit)

    tket_arch = get_tk_architecture(arch)
    unit = CompilationUnit(circuit, [ConnectivityPredicate(tket_arch)])
    passes = SequencePass([
        PlacementPass(GraphPlacement(tket_arch)),
        RoutingPass(tket_arch),
    ])
    passes.apply(unit)
    outcirc = make_routed_circ_simple(unit.circuit, unit.final_map)
    Transform.RebaseToTket().apply(outcirc)
    Transform.RebaseToRzRx().apply(outcirc)
    return outcirc


def simplify_cowtan(circuit: pytket.Circuit):
    outcirc = circuit
    PauliSimp().apply(outcirc)
    Transform.RebaseToTket().apply(outcirc)
    Transform.RebaseToRzRx().apply(outcirc)
    return outcirc


def simplify_cowtan_route_tket(circuit: pytket.Circuit, arch: Architecture):
    tket_arch = get_tk_architecture(arch)
    unit = CompilationUnit(circuit, [ConnectivityPredicate(tket_arch)])
    passes = SequencePass([
        PauliSimp(),
        PlacementPass(GraphPlacement(tket_arch)),
        RoutingPass(tket_arch),
    ])
    passes.apply(unit)
    outcirc = make_routed_circ_simple(unit.circuit, unit.final_map)
    Transform.RebaseToTket().apply(outcirc)
    Transform.RebaseToRzRx().apply(outcirc)
    return outcirc


def make_routed_circ_simple(circuit: pytket.Circuit, final_map):
    inv_map = {v: k for k, v in final_map.items()}
    outcirc = pytket.Circuit(circuit.n_qubits)
    for cmd in circuit:
        if isinstance(cmd, pytket.circuit.Command):
            tmp = list(map(lambda node: pytket.Qubit("q", node.index), cmd.qubits))
            remaped_qubits = list(map(lambda node: inv_map[node], cmd.qubits))
            outcirc.add_gate(cmd.op, remaped_qubits)
    return outcirc


def get_tk_architecture(architecture):
    coupling_graph = [e for e in architecture.graph.edges()]
    return pytket.routing.Architecture(coupling_graph)


def sample_evaluation(circ_in, zx_polynomial, arch: Architecture, meta_data: MetaData, n_reps=5, scenario="random"):
    if meta_data.algorithm == "divide_fast" or meta_data.algorithm == "divide_gauss":
        zx_polynomial = zx_polynomial.simplified()
        zx_polynomial = ZXPhasePoly.from_pauli_op(zx_polynomial, arch)

        circ_in = repeat_circuit(circ_in.copy(), n_reps=n_reps)
        start = time.time()
        circ_out = synth_divide_and_conquer(zx_polynomial,
                                            n_reps=n_reps,
                                            gaussian_step=meta_data.algorithm == "divide_gauss")
        time_diff = time.time() - start
        return get_circuit_stats(circ_in, circ_out, zx_polynomial.architecture, meta_data, time_diff, scenario=scenario)
    elif meta_data.algorithm == "pauli_opt":
        phase_circuit = zx_polynomial.simplified()
        topology = architecture_to_topology(arch)
        opt = OptimizedPhaseCircuit(phase_circuit, topology, cx_block=3, circuit_rep=n_reps)
        circ_in = repeat_circuit(circ_in, n_reps=n_reps)
        start = time.time()
        opt.anneal(num_iters=4800, schedule=("linear", 1.5, 0.1))
        time_diff = time.time() - start
        circ_out = opt.to_qiskit()
        return get_circuit_stats(circ_in, circ_out, arch, meta_data, time_diff, scenario=scenario,
                                 default_cx_count=opt.cx_count)
    elif meta_data.algorithm == "pyzx":
        topology = architecture_to_topology(arch)
        circ_out = repeat_circuit(zx_polynomial.simplified().to_qiskit(topology), n_reps=n_reps)
        circ_in = repeat_circuit(circ_in, n_reps=n_reps)

        start = time.time()
        circ_out = simplify_pyzx_route_tket(circ_out, arch)
        time_diff = time.time() - start
        return get_circuit_stats(circ_in, circ_out, arch, meta_data, time_diff, scenario=scenario)
    elif meta_data.algorithm == "tket":
        topology = architecture_to_topology(arch)
        zx_polynomial = repeat_circuit(zx_polynomial.simplified().to_qiskit(topology), n_reps=n_reps)
        circ_in = repeat_circuit(circ_in, n_reps=n_reps)

        zx_polynomial = qiskit_to_tket(zx_polynomial)
        start = time.time()
        circ_out = simplify_cowtan_route_tket(zx_polynomial, arch)
        time_diff = time.time() - start
        return get_circuit_stats(circ_in, circ_out, arch, meta_data, time_diff, scenario=scenario)
    elif meta_data.algorithm == "pure_tket":
        topology = architecture_to_topology(arch)
        zx_polynomial = repeat_circuit(zx_polynomial.simplified().to_qiskit(topology), n_reps=n_reps)
        circ_in = repeat_circuit(circ_in, n_reps=n_reps)

        zx_polynomial = qiskit_to_tket(zx_polynomial)
        start = time.time()
        circ_out = simplify_cowtan(zx_polynomial)
        time_diff = time.time() - start
        return get_circuit_stats(circ_in, circ_out, arch, meta_data, time_diff, scenario=scenario)
    elif meta_data.algorithm == "pure_pyzx":
        topology = architecture_to_topology(arch)
        circ_out = repeat_circuit(zx_polynomial.simplified().to_qiskit(topology), n_reps=n_reps)
        circ_in = repeat_circuit(circ_in, n_reps=n_reps)

        start = time.time()
        circ_out = simplify_pyzx(circ_out)
        time_diff = time.time() - start
        return get_circuit_stats(circ_in, circ_out, arch, meta_data, time_diff, scenario=scenario)
    else:
        raise Exception(f"Unknown method: {meta_data.algorithm}")


def generate_max_cut_qaoa(n_vertices, p_edge, arch_name):
    arch = create_architecture(arch_name, n_qubits=n_vertices)

    G = nx.erdos_renyi_graph(n_vertices, p_edge)

    phase_circuit = PhaseCircuit(n_vertices)

    for i in range(n_vertices):
        phase_circuit >>= X(pi / 2) @ {i}

    for u, v in G.edges(data=False):
        phase_circuit >>= Z(pi / 2) @ {v, u}

    qc = ZXPolyRegion.from_zx_polynomial(ZXPhasePoly.from_pauli_op(phase_circuit, arch)).to_qiskit()
    return qc, phase_circuit, arch


def compare_max_cut(n_qubits, p_edges, n_layer_reps, architectures, n_reps, algorithms,
                    df_name="evaluations/data/maxcut.csv"):
    df = pd.DataFrame()
    for n_qubit in n_qubits:
        for p_edge in p_edges:
            for n_layer_rep in n_layer_reps:
                for arch_name in architectures:
                    for n_rep in range(n_reps[0], n_reps[1]):
                        circ_in, phase_circ, arch = generate_max_cut_qaoa(n_qubit, p_edge, arch_name)
                        for algorithm in algorithms:
                            meta_data = MaxCutMetadata(n_rep, n_qubit, algorithm, arch_name, p_edge, n_layer_rep)
                            print(meta_data)
                            column = sample_evaluation(circ_in.copy(), phase_circ.cloned(), arch, meta_data,
                                                       scenario="maxcut", n_reps=n_layer_rep)
                            df = df.append(column, ignore_index=True)
                            df.to_csv(df_name)


def compare_random_limit_legs(n_qubits, n_pgs, n_reps, algorithms, architectures, max_qubits,
                              df_name="evaluations/data/random_limited.csv"):
    df = pd.DataFrame()
    for n_qubit in n_qubits:
        for n_pg in n_pgs:
            for arch_name in architectures:
                for max_qubit in max_qubits:
                    for n_rep in range(n_reps[0], n_reps[1]):
                        qc, phase_poly, arch = generate_random_zx_polynomial_limit_legs(n_qubit, n_pg, arch_name,
                                                                                        max_legs=max_qubit)
                        for algorithm in algorithms:
                            meta_data = RandomMetaData(n_rep, n_qubit, n_pg, algorithm, arch_name, 1.0)
                            print(meta_data)
                            column = sample_evaluation(qc.copy(), phase_poly, arch, meta_data,
                                                       scenario="random_limited")
                            df = df.append(column, ignore_index=True)
                            df.to_csv(df_name)
    df.to_csv(df_name)


def compare_random_zx_polys(n_qubits, n_pgs, n_reps, algorithms, max_legs, architectures,
                            df_name="evaluations/data/random.csv"):
    df = pd.DataFrame()
    for n_qubit in n_qubits:
        for n_pg in n_pgs:
            for max_leg in max_legs:
                for arch_name in architectures:
                    for n_rep in range(n_reps[0], n_reps[1]):
                        qc, phase_poly, arch = generate_random_zx_polynomial_limit_legs(n_qubit, n_pg,
                                                                                        arch_name, max_leg)
                        for algorithm in algorithms:
                            meta_data = RandomMetaData(n_rep, n_qubit, n_pg, algorithm, arch_name, max_leg)
                            print(meta_data)
                            column = sample_evaluation(qc.copy(), phase_poly, arch, meta_data, scenario="random")
                            df = df.append(column, ignore_index=True)
                            df.to_csv(df_name)
    df.to_csv(df_name)


def main(argv):
    if len(argv) < 2:
        raise Exception("Expected type of execution as program argument")
    elif argv[1] == "-r":
        n_rep_start = int(argv[2])
        n_rep_end = int(argv[3])
        print("Random execution")
        n_qubits = [4, 9, 16]
        n_pgs = [16, 32, 64, 128]
        max_legs = [3, 4]
        architectures = [FULLY_CONNECTED, LINE, SQUARE]
        n_reps = [n_rep_start, n_rep_end]  # 1it ~12h
        algorithms = ["divide_fast", "pauli_opt", "pyzx", "tket"]
        start = time.time()
        compare_random_zx_polys(n_qubits, n_pgs, n_reps, algorithms, max_legs, architectures,
                                df_name=f"evaluations/data/random_{n_rep_start}_{n_rep_end}.csv")
        print("Done: ", time.time() - start)  # 1 execution ~1547s
    elif argv[1] == "-rl":
        n_rep_start = int(argv[2])
        n_rep_end = int(argv[3])
        print("Random execution")
        n_qubits = [9, 16, 25]
        n_pgs = [10, 30, 50, 70, 90, 110, 130]
        architectures = [SQUARE]
        max_qubits = [4]
        n_reps = [n_rep_start, n_rep_end]
        algorithms = ["divide_fast", "pauli_opt"]
        start = time.time()  # 20 it ~134s
        compare_random_limit_legs(n_qubits, n_pgs, n_reps, algorithms, architectures, max_qubits,
                                  df_name=f"evaluations/data/limited_random_{n_rep_start}_{n_rep_end}.csv")
        print("Done: ", time.time() - start)
    elif argv[1] == "-mc":
        n_rep_start = int(argv[2])
        n_rep_end = int(argv[3])
        n_qubits = [4, 9, 16]
        n_layer_reps = [1, 3, 5, 7]
        architectures = [FULLY_CONNECTED, SQUARE, LINE]
        n_reps = [n_rep_start, n_rep_end]  # 1rep ~15h
        algorithms = ["divide_fast", "pauli_opt", "pyzx", "tket"]
        p_edges = [0.5, 0.7, 0.9]
        start = time.time()
        compare_max_cut(n_qubits, p_edges, n_layer_reps, architectures, n_reps, algorithms,
                        df_name=f"evaluations/data/maxcut_{n_rep_start}_{n_rep_end}.csv")
        print("Done: ", time.time() - start)  # 1 it ~ 6h
    elif argv[1] == "-gf":
        n_rep_start = int(argv[2])
        n_rep_end = int(argv[3])
        print("Gauss Fast comparison")
        n_qubits = [4, 5, 6]
        n_pgs = [10, 30, 50, 70, 90]
        architectures = [LINE, CIRCLE, FULLY_CONNECTED]
        max_qubits = [4]
        n_reps = [n_rep_start, n_rep_end]
        algorithms = ["divide_fast", "divide_gauss"]
        start = time.time()
        compare_random_limit_legs(n_qubits, n_pgs, n_reps, algorithms, architectures, max_qubits,
                                  df_name=f"evaluations/data/gauss_fast_{n_rep_start}_{n_rep_end}.csv")
        print("Done: ", time.time() - start)  # 1 it ~ 6h
    elif argv[1] == "-ppt":
        n_rep_start = int(argv[2])
        n_rep_end = int(argv[3])
        print("Random execution FC Architecture")
        n_qubits = [4, 9, 16]
        n_pgs = [16, 32, 64, 128]
        max_legs = [3, 4]
        architectures = [FULLY_CONNECTED]
        n_reps = [n_rep_start, n_rep_end]  # 1it ~12h
        algorithms = ["divide_fast", "pauli_opt", "pyzx", "tket", "pure_tket", "pure_pyzx"]
        start = time.time()
        compare_random_zx_polys(n_qubits, n_pgs, n_reps, algorithms, max_legs, architectures,
                                df_name=f"evaluations/data/random_pure_{n_rep_start}_{n_rep_end}.csv")
        print("Done: ", time.time() - start)  # 1 execution ~1547s


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)
