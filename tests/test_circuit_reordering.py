import unittest

from pyzx import Mat2

from ZXPhasePoly import ZXPhasePoly
from architecture import create_architecture, LINE
from optimization.optimisation import optimize_zx_poly
from synth.divide_utils import ParityRegion, ZXPolyRegion
from synth.synth_divide import synth_divide_conquer_step, region_to_qiskit, synth_divide_and_conquer, \
    sort_zx_polynomial, sort_zx_region
from synth.utils import get_matrix, get_phases
from utils import *


class TestCircuitSynth(unittest.TestCase):

    def test_random_zx_poly_swap(self):
        for _ in range(100):
            print("At: ", _)
            zx_poly_ = generate_random_phase_poly(4, 20)
            zx_poly = sort_zx_polynomial(zx_poly_)
            circ_out = tket_to_qiskit(zx_poly.to_tket())
            self.assertTrue(verify_equality(tket_to_qiskit(zx_poly_.to_tket()), circ_out))

    def test_specific_circuit_zx_poly_swap(self):
        for _ in range(100):
            print("At: ", _)
            circ = generate_random_phase_circuit(5, 25)
            zx_poly = ZXPhasePoly.from_circuit(qiskit_to_tket(circ))
            zx_poly = sort_zx_polynomial(zx_poly)
            self.assertTrue(verify_equality(tket_to_qiskit(zx_poly.to_tket()), circ))

    def test_sort_zx_region(self):
        for _ in range(100):
            print("At: ", _)
            zx_poly_ = generate_random_phase_poly(4, 20)
            matrix = get_matrix(zx_poly_)
            phases = get_phases(zx_poly_)

            region = ZXPolyRegion(matrix, phases, zx_poly_.n_qubits, zx_poly_.architecture)
            region = sort_zx_region(region)
            circ_out = region_to_qiskit([region])
            self.assertTrue(verify_equality(tket_to_qiskit(zx_poly_.to_tket()), circ_out))
