import unittest

from pauliopt.topologies import Topology
from pyzx import full_reduce

from ZXPhasePoly import ZXPhasePoly
from architecture import LINE, SQUARE, FULLY_CONNECTED
from evaluations import generate_random_zx_polynomial
from synth.divide_utils import ZXPolyRegion
from synth.utils import get_phases, get_matrix
from utils import *


class TestCircuitConversion(unittest.TestCase):
    def test_circuit_parsing(self):
        for _ in range(10):
            print("At: ", _)
            circ = generate_random_phase_circuit(5, 100)
            zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ))
            circ_out = zx_poly.to_tket()

            self.assertTrue(verify_equality(circ, tket_to_qiskit(circ_out)), msg=f"At {_}")

    def test_pure_rotation_gate_circuit(self):
        circ = pytket.Circuit(3)
        circ.Rz(12.5, 0)
        circ.Rx(0.001, 1)
        circ.CX(2, 0)
        circ.Rz(13.333333, 0)
        circ.CX(0, 1)
        circ.Rx(0.001, 2)
        zx_poly = ZXPhasePoly.from_circuit(circ)
        circ_out = zx_poly.to_tket()
        print(tket_to_qiskit(circ))
        print(tket_to_qiskit(circ_out))

        self.assertTrue(verify_equality(tket_to_qiskit(circ), tket_to_qiskit(circ_out)))

    def test_pure_cnot_circuit(self):
        circ = pytket.Circuit(3)
        circ.CX(0, 1)
        circ.CX(1, 2)
        circ.CX(2, 1)
        circ.CX(0, 2)
        zx_poly = ZXPhasePoly.from_circuit(circ)
        circ_out = zx_poly.to_tket()
        self.assertTrue(verify_equality(tket_to_qiskit(circ), tket_to_qiskit(circ_out)))

    def test_pauli_opt_conversion(self):
        for qubits in [4, 9]:
            for gadgets in [50]:
                for arch_name in [SQUARE, LINE, FULLY_CONNECTED]:
                    print(arch_name, gadgets, qubits)
                    for _ in range(10):
                        print(_)
                        qc, phase_circuit, arch = generate_random_zx_polynomial(qubits, gadgets, arch_name)

                        zx_poly = ZXPhasePoly.from_pauli_op(phase_circuit.cloned(), arch)
                        region = ZXPolyRegion(get_matrix(zx_poly), get_phases(zx_poly), zx_poly.n_qubits,
                                              zx_poly.architecture)
                        self.assertTrue(verify_equality(zx_poly.to_qiskit(), region.to_qiskit()))
