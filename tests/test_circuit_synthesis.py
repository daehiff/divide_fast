import unittest

from pyzx import Mat2

from ZXPhasePoly import ZXPhasePoly
from architecture import create_architecture, LINE
from optimization.optimisation import optimize_zx_poly
from synth.divide_utils import ParityRegion, ZXPolyRegion
from synth.synth_divide import synth_divide_conquer_step, region_to_qiskit, synth_divide_and_conquer
from synth.utils import get_matrix, get_phases
from utils import *


class TestCircuitSynth(unittest.TestCase):
    def test_specific_circuit_synthesis_random_zx_poly_line(self):
        for gauss in [False, True]:
            print("Gauss: ", gauss)
            for _ in range(5):
                print(_)
                arch = create_architecture(LINE, n_qubits=4)
                zx_poly = generate_random_phase_poly(4, 20, arch=arch)
                circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=gauss)
                self.assertTrue(verify_equality(tket_to_qiskit(zx_poly.to_tket()), circ_out))

    def test_specific_circuit_synthesis_random_zx_poly(self):
        for gauss in [False, True]:
            print("Gauss: ", gauss)
            for _ in range(5):
                print(_)

                zx_poly = generate_random_phase_poly(4, 20)
                circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=gauss)
                self.assertTrue(verify_equality(tket_to_qiskit(zx_poly.to_tket()), circ_out))

    def test_specific_circuit_synthesis_random_circ_line(self):
        for gauss in [False, True]:
            print("Gauss: ", gauss)
            for _ in range(5):
                print(_)
                arch = create_architecture(LINE, n_qubits=4)
                circ = generate_random_phase_circuit(4, 10)

                zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ), architecture=arch)
                circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=gauss)
                self.assertTrue(verify_equality(circ, circ_out))

    def test_specific_circuit_synthesis_random_circ(self):
        for gauss in [False, True]:
            print("Gauss: ", gauss)
            for _ in range(5):
                print(_)
                circ = generate_random_phase_circuit(4, 10)

                zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ))
                circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=gauss)
                self.assertTrue(verify_equality(circ, circ_out))

    def test_specific_circ_divide_concquer(self):
        circ = pytket.Circuit(3)
        circ.Rz(12.5, 0)
        circ.Rx(0.001, 1)
        circ.CX(2, 0)
        circ.Rz(13.333333, 0)
        circ.CX(0, 1)
        circ.Rx(0.001, 2)

        zx_poly = ZXPhasePoly.from_circuit(circ)
        circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=False)

        print(tket_to_qiskit(circ))
        print(circ_out)
        self.assertTrue(verify_equality(tket_to_qiskit(circ), circ_out))

        zx_poly = ZXPhasePoly.from_circuit(circ)
        circ_out = synth_divide_and_conquer(zx_poly, gaussian_step=True)

        print(tket_to_qiskit(circ))
        print(circ_out)
        self.assertTrue(verify_equality(tket_to_qiskit(circ), circ_out))
