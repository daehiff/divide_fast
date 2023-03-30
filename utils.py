import random
from fractions import Fraction

import numpy as np
import pytket
import pyzx
from pytket._tket.circuit import Command, OpType
from qiskit import QuantumCircuit
from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str

from pytket.extensions.pyzx import tk_to_pyzx, pyzx_to_tk
from pytket.extensions.qiskit.qiskit_convert import tk_to_qiskit, qiskit_to_tk

import pyzx as zx
from qiskit.quantum_info import Operator, Statevector

CLIFFORD_T_SET = {"h", "cx", "x", "y", "z", "s", "t", "sdg", "tdg"}


def qiskit_to_tket(circuit: QuantumCircuit) -> pytket.Circuit:
    return qiskit_to_tk(circuit)


def tket_to_qiskit(circuit: pytket.Circuit) -> QuantumCircuit:
    return tk_to_qiskit(circuit)


def pyzx_to_tket(circuit: pyzx.Circuit) -> pytket.Circuit:
    return pyzx_to_tk(circuit)


def tket_to_pyzx(circuit: pytket.Circuit) -> pyzx.Circuit:
    return tk_to_pyzx(circuit, denominator_limit=np.inf)


def qiskit_pyzx(circ: QuantumCircuit) -> zx.Circuit:
    return tket_to_pyzx(qiskit_to_tket(circ))


def pyzx_to_qiskit(circuit: zx.Circuit) -> QuantumCircuit:
    return tket_to_qiskit(pyzx_to_tket(circuit))


def generate_random_phase_circuit(qubit_count, gate_count, p_cx=None, p_rx=None, p_rz=None):
    num = 0.0
    rest = 1.0
    if p_cx is None:
        num += 1.0
    else:
        rest -= p_cx
    if p_rx is None:
        num += 1.0
    else:
        rest -= p_rx
    if p_rz is None:
        num += 1.0
    else:
        rest -= p_rz

    if rest < 0:
        raise ValueError("Probabilities are >1.")

    if p_cx is None:
        p_cx = rest / num
    if p_rz is None:
        p_rz = rest / num
    if p_rx is None:
        p_rx = rest / num

    circ = pytket.Circuit(qubit_count)
    minimal_rotations = [Fraction(1, 1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)]
    for _ in range(gate_count):
        r = random.random()
        target = random.randint(0, qubit_count - 1)
        if r > 1 - p_cx:
            control = random.choice([i for i in range(qubit_count) if i != target])
            circ.CX(control, target)
        elif r > 1 - p_cx - p_rz:
            phase = random.choice(minimal_rotations)
            circ.Rz(phase, target)
        else:
            phase = random.choice(minimal_rotations)
            circ.Rx(phase, target)
    return tket_to_qiskit(circ)


def generate_random_circuit(qubit_count, gate_count, p_t=None, p_s=None, p_hsh=None, p_cnot=None) -> QuantumCircuit:
    circuit = zx.Circuit.from_graph(zx.generate.cliffordT(qubit_count, gate_count,
                                                          p_t=p_t,
                                                          p_s=p_s,
                                                          p_hsh=p_hsh,
                                                          p_cnot=p_cnot)).to_basic_gates().split_phase_gates()
    circuit = pyzx_to_qiskit(circuit)
    for gate, _, _ in circuit.data:
        assert gate.name in CLIFFORD_T_SET
    return circuit


def rebase_zx_decomposition(circuit: pytket.Circuit) -> pytket.Circuit:
    circuit_out = pytket.Circuit(circuit.n_qubits)
    for gate in circuit:
        if isinstance(gate, Command):

            if str(gate.op) == "H":
                idx = gate.qubits[0].index[0]
                circuit_out.Rz(1.0 / 2.0, idx)
                circuit_out.Rx(1.0 / 2.0, idx)
                circuit_out.Rz(1.0 / 2.0, idx)
            elif str(gate.op) == "S":
                idx = gate.qubits[0].index[0]
                circuit_out.Rz(1.0 / 2.0, idx)
            elif str(gate.op) == "T":
                idx = gate.qubits[0].index[0]
                circuit_out.Rz(1.0 / 4.0, idx)
            elif str(gate.op) == "CX":
                idx_c = gate.qubits[0].index[0]
                idx_t = gate.qubits[1].index[0]
                circuit_out.CX(idx_c, idx_t)
            elif str(gate.op) == "CZ":
                idx_c = gate.qubits[0].index[0]
                idx_t = gate.qubits[1].index[0]
                circuit_out.Rz(1.0 / 2.0, idx_t)
                circuit_out.Rx(1.0 / 2.0, idx_t)
                circuit_out.Rz(1.0 / 2.0, idx_t)

                circuit_out.CX(idx_c, idx_t)

                circuit_out.Rz(1.0 / 2.0, idx_t)
                circuit_out.Rx(1.0 / 2.0, idx_t)
                circuit_out.Rz(1.0 / 2.0, idx_t)
            else:
                print("Unknown Gate: ", gate.op)
        else:
            print("Error!, no instance of Command")
            return None
    return circuit_out


def verify_equality(qc_in: QuantumCircuit, qc_out: QuantumCircuit):
    """
    Verify the equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    return Statevector.from_instruction(qc_in) \
        .equiv(Statevector.from_instruction(qc_out))


def verify_equality_(qc_in: pytket.Circuit, qc_out: pytket.Circuit):
    """
    Verify Equality up to a global phase
    :param qc_in:
    :param qc_out:
    :return:
    """
    return Statevector.from_instruction(tket_to_qiskit(qc_in)) \
        .equiv(Statevector.from_instruction(tket_to_qiskit(qc_out)))


def parity_to_zx(par, type):
    out = []
    for p in par:
        if p == '1':
            out.append(type)
        else:
            out.append(0.0)
    return np.asarray(out)


def generate_random_phase_poly(n_qubits, n_gadgets, arch=None):
    parities = []

    if n_qubits < 64:
        while len(parities) < n_gadgets:
            parities.append(np.random.randint(1, 2 ** n_qubits))
    else:
        while len(parities) < n_gadgets:
            parities.append("".join(np.random.choice(["0", "1"], n_qubits, replace=True)))

    parities = [("{0:{fill}" + str(n_qubits) + "b}").format(integer, fill='0', align='right') for integer in parities]
    p_types = [random.randint(1, 2) for _ in range(len(parities))]
    matrix = np.asarray([parity_to_zx(par, type) for par, type in zip(parities, p_types)]).T
    phases = [random.choice([0.5, 0.25, 0.125]) for _ in range(len(parities))]
    if arch is None:
        from architecture import create_architecture
        from architecture import FULLY_CONNECTED
        arch = create_architecture(FULLY_CONNECTED, n_qubits=n_qubits)
    from ZXPhasePoly import ZXPhasePoly
    from pyzx import Mat2
    return ZXPhasePoly.from_matrix(matrix, phases, list(range(len(parities))), arch, n_qubits, Mat2.id(n_qubits))


def clamp(phase):
    new_phase = phase % 2
    if new_phase > 1:
        return new_phase - 2
    return phase
