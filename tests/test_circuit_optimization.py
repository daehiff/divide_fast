import unittest

from ZXPhasePoly import ZXPhasePoly
from optimization.optimisation import optimize_zx_poly
from utils import *


class TestCircuitOptimization(unittest.TestCase):
    def test_circuit_optimization(self):
        """
        Random testing to generate a phase circuit and oprimize it using our method
        """
        for _ in range(10):
            print("At: ", _)
            circ = generate_random_phase_circuit(5, 100)
            zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ))
            zx_poly = optimize_zx_poly(zx_poly)
            circ_out = zx_poly.to_tket()
            self.assertTrue(verify_equality_(qiskit_to_tket(circ), circ_out), msg=f"At {_}")

    def test_specific_circuit_optimization(self):
        circ = QuantumCircuit(5)
        circ.rz(0.052, 1)
        circ.rz(0.10, 1)
        circ.rx(np.pi / 4.0, 2)
        circ.cx(1, 3)
        circ.rx(np.pi / 3.0, 3)
        zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ))
        zx_poly = optimize_zx_poly(zx_poly)
        circ_out = zx_poly.to_tket()
        print(circ)
        print(tket_to_qiskit(circ_out))
        self.assertTrue(verify_equality_(qiskit_to_tket(circ), circ_out))

    def test_rotation_gate_circuit(self):
        circ = pytket.Circuit(3)
        circ.Rz(12.5, 0)
        circ.Rx(0.001, 1)
        circ.CX(2, 0)
        circ.Rz(13.333333, 0)
        circ.CX(0, 1)
        circ.Rx(0.001, 2)
        zx_poly = ZXPhasePoly.from_circuit(circ)
        zx_poly = optimize_zx_poly(zx_poly)
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
        zx_poly = optimize_zx_poly(zx_poly)
        circ_out = zx_poly.to_tket()
        self.assertTrue(verify_equality(tket_to_qiskit(circ), tket_to_qiskit(circ_out)))
