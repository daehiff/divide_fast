import pytket
import pyzx
from matplotlib import pyplot as plt
from pauliopt.phase import PhaseCircuit, PhaseGadget
from pauliopt.topologies import Topology
from pytket import OpType
from pytket._tket.circuit import PauliExpBox
from pytket._tket.pauli import Pauli
from pytket._tket.transform import Transform
from pytket.utils import Graph
from pyzx import Mat2
from pyzx.circuit import ZPhase, XPhase
from pyzx.circuit.gates import ParityPhase

from steiner import rec_steiner_gauss
from utils import *
import numpy as np
from architecture import create_architecture, FULLY_CONNECTED, Architecture


class RemainingCNOTTracker:
    def __init__(self):
        self.remaining_cnots = []

    def row_add(self, control, target):
        self.remaining_cnots.append((control, target))

    def col_add(self, control, target):
        self.remaining_cnots.append((target, control))


def matrix_column_to_parity(matrix, phases, col_idx):
    type_str = "x" if 1 in matrix[:, col_idx] else "z"
    par_i = "".join([str(i) for i in list(np.where(matrix[:, col_idx] == 0.0, 0, 1))])
    return {type_str: (par_i, phases[col_idx])}


def update_circuit(_circ: pytket.Circuit, command_: Command) -> pytket.Circuit:
    if command_.op.type == OpType.Rz:
        _circ.Rz(command_.op.params[0], command_.qubits[0].index[0])
    elif command_.op.type == OpType.Rx:
        _circ.Rx(command_.op.params[0], command_.qubits[0].index[0])
    elif command_.op.type == OpType.CX:
        idx_c = command_.qubits[0].index[0]
        idx_t = command_.qubits[1].index[0]
        _circ.CX(idx_c, idx_t)
    else:
        raise Exception(f"Unknown Gate: {command_.op.type}. Expected Rx, Rz or CX")
    return _circ


def find_pure_regions(circuit: pytket.Circuit) -> [pytket.Circuit]:
    pure_regions = []
    current_gate = None
    current_circ = pytket.Circuit(circuit.n_qubits)
    for command in circuit:
        if isinstance(command, Command):
            gate_type = command.op.type

            if current_gate is None:
                if gate_type != OpType.CX:
                    current_gate = gate_type
                current_circ = update_circuit(current_circ, command)
            elif gate_type == OpType.CX or current_gate == gate_type:
                current_circ = update_circuit(current_circ, command)
            else:
                pure_regions.append((current_gate, current_circ))
                current_gate = gate_type
                current_circ = pytket.Circuit(circuit.n_qubits)
                current_circ = update_circuit(current_circ, command)
        else:
            raise Exception("Unexpected type in gate")
    if current_gate is not None:
        pure_regions.append((current_gate, current_circ))
    else:
        if len(pure_regions) > 0:
            pure_regions.append((pure_regions[-1][0], current_circ))
        else:
            pure_regions.append((OpType.Rx, current_circ))

    circ_ = pytket.Circuit(circuit.n_qubits)
    for gate, region in pure_regions:
        for command in region:
            if isinstance(command, Command):
                circ_ = update_circuit(circ_, command)
            else:
                raise Exception("Unexpected Type in Gate")
    assert verify_equality_(circuit, circ_)

    return pure_regions


def parse_zx_poly(circuit: pytket.Circuit, architecture=None):
    zx_phases = []

    current_parities_x = Mat2.id(circuit.n_qubits)
    current_parities_z = Mat2.id(circuit.n_qubits)
    for gate in circuit:
        parse_gates(gate, current_parities_x, current_parities_z, zx_phases)

    if architecture is None:
        architecture = create_architecture(FULLY_CONNECTED, n_qubits=circuit.n_qubits)

    return ZXPhasePoly(zx_phases, current_parities_z.inverse(), circuit.n_qubits, architecture)


def parse_gate_2(gate, current_parities_x, current_parities_z, zx_phases):
    pass
    return current_parities_x, current_parities_z, zx_phases


def parse_gates_1(gate, current_parities_x, current_parities_z, zx_phases):
    if gate.name in ["CNOT", "CX"]:
        current_parities_z.row_add(gate.control, gate.target)
        current_parities_x.row_add(gate.target, gate.control)
    elif isinstance(gate, ZPhase):
        parity = current_parities_z.data[gate.target]
        parity = "".join([str(par) for par in parity])
        # Add the T rotation to the phases
        zx_phase = {
            "z": (parity, gate.phase)
        }
        zx_phases.append(zx_phase)
    elif isinstance(gate, XPhase):
        parity = current_parities_x.data[gate.target]
        parity = "".join([str(par) for par in parity])
        zx_phase = {
            "x": (parity, gate.phase)
        }
        zx_phases.append(zx_phase)
    else:
        raise Exception("Gate not supported! The following gate gets ignored: ", gate.name)
    return current_parities_x, current_parities_z, zx_phases


def parse_gates(gate: Command, current_parities_x: Mat2, current_parities_z: Mat2, phases):
    if gate.op.type == pytket.OpType.CX:
        control = gate.args[0].index[0]
        target = gate.args[1].index[0]
        current_parities_z.row_add(control, target)
        current_parities_x.row_add(target, control)
    elif gate.op.type == pytket.OpType.Rz:
        parity = current_parities_z.data[gate.args[0].index[0]]
        parity = "".join([str(par) for par in parity])
        # Add the T rotation to the phases
        zx_phase = {
            "z": (parity, clamp(gate.op.params[0]))
        }
        phases.append(zx_phase)
    elif gate.op.type == pytket.OpType.Rx:
        parity = current_parities_x.data[gate.args[0].index[0]]
        parity = "".join([str(par) for par in parity])

        zx_phase = {
            "x": (parity, clamp(gate.op.params[0]))
        }
        phases.append(zx_phase)
    else:
        raise Exception("Gate not supported: ", gate.name)


class ZXPhasePoly:

    def __init__(self, zx_phases: list, global_parities: Mat2, n_qubits: int, architecture: Architecture):
        self.zx_phases = zx_phases
        self.n_qubits = n_qubits
        self.global_parities = global_parities
        self.architecture = architecture

    @staticmethod
    def from_pauli_op(circ: PhaseCircuit, arch: Architecture):
        zx_phases = []

        for pgs in circ:
            assert isinstance(pgs, PhaseGadget)
            parity = "".join(["1" if i in list(pgs.qubits) else "0" for i in range(circ.num_qubits)])
            angle = pgs.angle
            basis = pgs.basis.lower()
            zx_phases.append({basis: (parity, float(angle) / np.pi)})
        return ZXPhasePoly(zx_phases, Mat2.id(circ.num_qubits), circ.num_qubits, arch)

    def get_phase(self, phase_idx):
        assert phase_idx < len(self.zx_phases)
        poly = self.zx_phases[phase_idx]
        assert isinstance(poly, dict)
        if "z" in poly:
            return poly["z"][1]
        else:
            return poly["x"][1]

    def get_parity(self, phase_idx):
        assert phase_idx < len(self.zx_phases)
        poly = self.zx_phases[phase_idx]
        assert isinstance(poly, dict)
        if "z" in poly:
            return poly["z"][0]
        else:
            return poly["x"][0]

    def get_rotation_tyoe(self, phase_idx):
        assert phase_idx < len(self.zx_phases)
        poly = self.zx_phases[phase_idx]
        assert isinstance(poly, dict)
        if "z" in poly:
            return "z"
        else:
            return "x"

    @staticmethod
    def from_circuit(circuit: pytket.Circuit, architecture=None):
        return parse_zx_poly(circuit, architecture=architecture)

    @staticmethod
    def from_matrix(matrix, phases, cols_to_use, architecture, n_qubits, global_parities):
        parities = [matrix_column_to_parity(matrix, phases, i) for i in cols_to_use]
        return ZXPhasePoly(parities, global_parities, n_qubits, architecture)

    def to_qiskit(self):
        return tket_to_qiskit(self.to_tket())

    def to_tket(self):
        circuit = pytket.Circuit(self.n_qubits)
        for zx_phase in self.zx_phases:
            for rotation_type, (parity, phase) in zx_phase.items():
                qubits = ([i for i, s in enumerate(parity) if s == '1'])
                if rotation_type == "z":
                    circuit.add_pauliexpbox(PauliExpBox([Pauli.Z] * len(qubits), phase), qubits)
                else:
                    circuit.add_pauliexpbox(PauliExpBox([Pauli.X] * len(qubits), phase), qubits)

        Transform.DecomposeBoxes().apply(circuit)

        cnots = RemainingCNOTTracker()
        rec_steiner_gauss(self.global_parities.copy(), self.architecture, y=cnots, full_reduce=True)
        for control, target in cnots.remaining_cnots:
            circuit.CX(control, target)

        return circuit


def partition2mat2(partition):
    return Mat2([[int(i) for i in parity] for parity in partition])


def mat22partition(m):
    return ["".join(str(i) for i in parity) for parity in m.data]
