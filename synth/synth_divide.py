import itertools
import random
import time

from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology
from pyzx import Mat2

from ZXPhasePoly import ZXPhasePoly
from architecture import Architecture
from optimization.optimisation import optimize_zx_poly
from utils import *
from .divide_utils import ZXPolyRegion, ParityRegion, get_distance_map
from .utils import get_matrix, get_phases, is_able_to_swap


def region_to_qiskit(region):
    assert len(region) >= 1, "Received empty region"
    circ = QuantumCircuit(region[0].n_qubits)
    for el in region:
        circ += el.to_qiskit()
    return circ


def pick_random_cnots(num_iters, n_qubits):
    assert num_iters <= n_qubits ** 2 - n_qubits
    cnots = [(c, t) for c in range(n_qubits) for t in range(n_qubits) if c != t]
    return random.sample(cnots, num_iters)


def optimize_region_gauss(par_left: ParityRegion, zx_region: ZXPolyRegion, par_right: ParityRegion, leg_chache):
    for ctrl in range(zx_region.n_qubits):
        for target in range(zx_region.n_qubits):
            if ctrl != target and \
                    get_effect(par_left, zx_region, par_right, ctrl, target, True, leg_chache) < -1:
                zx_region.update(ctrl, target)
                par_right.update(ctrl, target, "l")
                par_left.update(ctrl, target, "r")
    return par_left, zx_region, par_right


def optimize_region_all_qubits(par_left: ParityRegion, zx_region: ZXPolyRegion, par_right: ParityRegion, leg_chache):
    dist = get_distance_map(zx_region.architecture)
    for ctrl, target in (par_right.get_added_gates()):
        if get_effect(par_left, zx_region, par_right, ctrl, target, False, leg_chache) < -1 * dist[ctrl, target]:
            zx_region.update(ctrl, target)
            par_right.update(ctrl, target, "l")
            par_left.update(ctrl, target, "r")

    for ctrl, target in reversed(par_left.get_added_gates()):
        if get_effect(par_left, zx_region, par_right, ctrl, target, False, leg_chache) < -1 * dist[ctrl, target]:
            zx_region.update(ctrl, target)
            par_right.update(ctrl, target, "l")
            par_left.update(ctrl, target, "r")

    for ctrl in range(zx_region.n_qubits):
        for trg in range(zx_region.n_qubits):
            if ctrl != trg and \
                    get_effect(par_left, zx_region, par_right, ctrl, trg, False, leg_chache) < -1 * dist[ctrl, trg]:
                zx_region.update(ctrl, trg)
                par_right.update(ctrl, trg, "l")
                par_left.update(ctrl, trg, "r")
    return par_left, zx_region, par_right


def optimize_region(par_left: ParityRegion, zx_region: ZXPolyRegion, par_right: ParityRegion, leg_chache,
                    optimize_gauss=True):
    """
    Optimize a ZX Region: Iterate over the region until we don't see a negative effect
    Negative effect means, that we can pull out CNOTs to make the Region more effective
    :param par_left:
    :param zx_region:
    :param par_right:
    :return:
    """
    if optimize_gauss:
        return optimize_region_gauss(par_left, zx_region, par_right, leg_chache)
    else:
        return optimize_region_all_qubits(par_left, zx_region, par_right, leg_chache)


def get_effect(par_left: ParityRegion, zx: ZXPolyRegion, par_right: ParityRegion, ctrl, target, optimize_par,
               leg_chache):
    if optimize_par:
        effect_left = par_left.get_effect_local(ctrl, target, "r") \
                      + zx.get_effect_local(ctrl, target, leg_chache) \
                      + par_right.get_effect_local(ctrl, target, "l")
    else:
        effect_left = zx.get_effect_local(ctrl, target, leg_chache)
    return effect_left


def create_region_effect(par_left: ParityRegion, zx: ZXPolyRegion, par_right: ParityRegion, leg_chache,
                         optimize_par=True):
    """
    Compute the O(q^2) Matrix, that describes the effect of each cnot(i,j) at E_ij
    :param par_left:
    :param zx:
    :param par_right:
    :return:
    """
    effect = np.zeros((zx.n_qubits, zx.n_qubits))
    for ctrl in list(range(zx.n_qubits)):
        for target in list(range(zx.n_qubits)):
            if ctrl != target:
                effect[ctrl, target] = get_effect(par_left, zx, par_right, ctrl, target, optimize_par, leg_chache)
    return effect


def matching_legs(matrix, col, col_i):
    par_j = np.where(matrix[:, col_i] == 0.0, 'A', 1.0)
    par_i = np.where(matrix[:, col] == 0.0, 'B', 1.0)
    return np.sum(2 * (par_j == par_i) - 1)


def synth_divide_conquer_step(left: ParityRegion, zx_region: ZXPolyRegion, right: ParityRegion, leg_chache,
                              gaussian=True):
    if len(zx_region.matrix[1]) <= 2:
        left, zx_region, right = optimize_region(left, zx_region, right, leg_chache, optimize_gauss=gaussian)
        return [left, zx_region, right]
    else:
        left, zx_region, right = optimize_region(left, zx_region, right, leg_chache, optimize_gauss=gaussian)

        optimize_placement_matrix(zx_region.matrix, zx_region.angles)

        center_idx = int(zx_region.matrix.shape[1] / 2.0)
        zx_region_left = ZXPolyRegion(zx_region.matrix[:, :center_idx], zx_region.angles[:center_idx],
                                      zx_region.n_qubits, zx_region.architecture)
        zx_region_left = sort_zx_region(zx_region_left, asc=True)
        zx_region_right = ZXPolyRegion(zx_region.matrix[:, center_idx:], zx_region.angles[center_idx:],
                                       zx_region.n_qubits, zx_region.architecture)
        zx_region_right = sort_zx_region(zx_region_right, asc=False)

        center = ParityRegion(Mat2.id(left.n_qubits), left.n_qubits, left.architecture)
        sub_region_left = synth_divide_conquer_step(left, zx_region_left, center, leg_chache, gaussian)
        sub_region_right = synth_divide_conquer_step(center, zx_region_right, right, leg_chache, gaussian)

        return sub_region_left + sub_region_right[1:]


def architecture_to_topology(arch: Architecture):
    edge_list = sorted([list(e) for e in arch.graph.edge_set()])
    return Topology.from_dict(
        {
            "num_qubits": arch.n_qubits,
            "couplings": edge_list
        }
    )


def get_count(circuit: QuantumCircuit, type="two_qubit"):
    count_ops = dict(circuit.count_ops())
    if type == "two_qubit":
        count = 0
        if "cx" in count_ops:
            count += count_ops["cx"]
        if "swap" in count_ops:
            count += count_ops["swap"]
        if "cz" in count_ops:
            count += count_ops["cz"]
        return count
    elif type == "one_qubit":
        # TOOD does this work?
        count = 0
        if "rx" in count_ops:
            count += count_ops["rx"]
        if "rz" in count_ops:
            count += count_ops["rz"]
        return count


def nr_legs(matrix, row_i):
    par_i = np.where(matrix[:, row_i] == 0.0, 0.0, 1.0)
    return np.sum(par_i)


def compare(matrix, col_prev, col, col_next):
    return is_able_to_swap(matrix, col, col_next) and \
           matching_legs(matrix, col_prev, col) < matching_legs(matrix, col_prev, col_next)


def swap(matrix, phases, col_i, col_j):
    tmp = matrix[:, col_i].copy()
    matrix[:, col_i] = matrix[:, col_j]
    matrix[:, col_j] = tmp

    tmp_ = phases[col_i]
    phases[col_i] = phases[col_j]
    phases[col_j] = tmp_


def compare_sort(matrix, col, col_next):
    return is_able_to_swap(matrix, col, col_next) and \
           (nr_legs(matrix, col) > nr_legs(matrix, col_next))


def get_region_effect(matrix, region):
    effect = 0
    for col_i in region:
        for col_j in region:
            if np.sum(np.where(matrix[:, col_i] == 0.0, 0, 1.0)) == 1 or \
                    np.sum(np.where(matrix[:, col_j] == 0.0, 0.0, 1.0)) == 1:
                continue
            par_j = np.where(matrix[:, col_i] == 0.0, 'A', 1.0)
            par_i = np.where(matrix[:, col_j] == 0.0, 'B', 1.0)
            effect += np.sum(par_i == par_j)
    return effect


def find_best_center(matrix, dist=0.1):
    max_col = None
    max_effect = -np.inf
    lb = max(1, int(matrix.shape[1] / 2.0 - matrix.shape[1] * dist))
    ub = min(matrix.shape[1], int(matrix.shape[1] / 2.0 - matrix.shape[1] * dist) + 1)
    if lb == ub:
        ub += 1

    for col in range(lb, ub):
        effect_left = get_region_effect(matrix, list(range(col)))
        effect_right = get_region_effect(matrix, list(range(col, matrix.shape[1])))
        if effect_right + effect_left > max_effect:
            max_col = col
            max_effect = effect_right + effect_left
    return max_col


def sort_matrix(matrix, phases, asc=True):
    if asc:
        col_idx = 0
        while col_idx < matrix.shape[1]:
            col_idx_ = col_idx
            new_col_idx = col_idx_ + 1
            while new_col_idx < matrix.shape[1] and compare_sort(matrix, new_col_idx, col_idx_):
                swap(matrix, phases, new_col_idx, col_idx_)
                col_idx_ = new_col_idx
                new_col_idx += 1
            col_idx += 1
    else:
        col_idx = matrix.shape[1] - 1
        while col_idx >= 0:
            col_idx_ = col_idx
            new_col_idx = col_idx_ - 1
            while new_col_idx >= 0 and compare_sort(matrix, new_col_idx, col_idx_):
                swap(matrix, phases, new_col_idx, col_idx_)
                col_idx_ = new_col_idx
                new_col_idx -= 1
            col_idx -= 1


def optimize_placement_matrix(matrix, phases):
    col_idx = 1
    while col_idx < matrix.shape[1] - 1:
        prev_col_idx = col_idx - 1
        col_idx_ = col_idx
        new_col_idx = col_idx_ + 1
        while new_col_idx < matrix.shape[1] and compare(matrix, prev_col_idx, col_idx_, new_col_idx):
            swap(matrix, phases, new_col_idx, col_idx_)
            prev_col_idx = col_idx_
            col_idx_ = new_col_idx
            new_col_idx = col_idx_ + 1

        col_idx += 1


def sort_zx_region(zx_region: ZXPolyRegion, asc=False):
    matrix = zx_region.matrix
    phases = zx_region.angles
    sort_matrix(matrix, phases, asc=asc)
    optimize_placement_matrix(matrix, phases)
    zx_region.matrix = matrix
    zx_region.angles = phases
    return zx_region


def sort_zx_polynomial(zx_poly: ZXPhasePoly):
    matrix = get_matrix(zx_poly)
    phases = get_phases(zx_poly)
    sort_matrix(matrix, phases, asc=False)
    optimize_placement_matrix(matrix, phases)

    return ZXPhasePoly.from_matrix(matrix, phases, list(range(matrix.shape[1])),
                                   zx_poly.architecture,
                                   zx_poly.n_qubits,
                                   zx_poly.global_parities)


def get_non_matching_cnots(circuit: QuantumCircuit, architecture: Architecture):
    count = 0
    for instruction, qubits, b in circuit:
        if instruction.name == "cx" or instruction.name == "cnot":
            ctrl_idx = qubits[0]._index
            target_idx = qubits[1]._index
            if not target_idx in architecture.get_neighboring_qubits(ctrl_idx):
                count += 1
    return count


def repeat_circuit(zx_polynomial: QuantumCircuit, n_reps=1):
    circ = QuantumCircuit(zx_polynomial.num_qubits)
    for _ in range(n_reps):
        circ += zx_polynomial
    return circ


def synth_divide_and_conquer(zx_polynomial: ZXPhasePoly, gaussian_step=False, n_reps=1):
    start = time.time()
    left, region, right = synth_divide_and_conquer_(zx_polynomial, gaussian_step)
    print("Done: ", time.time() - start)
    region_out = [left] + region * n_reps + [right]
    return region_to_qiskit(region_out)


def synth_divide_and_conquer_(zx_polynomial: ZXPhasePoly, gaussian_step=False):
    zx_polynomial = optimize_zx_poly(zx_polynomial)
    zx_poly = zx_polynomial

    matrix = get_matrix(zx_poly)
    angels = get_phases(zx_poly)
    left = ParityRegion(Mat2.id(zx_poly.n_qubits), zx_poly.n_qubits, zx_poly.architecture)
    zx_region = ZXPolyRegion(matrix, angels, zx_poly.n_qubits, zx_poly.architecture)
    right = ParityRegion(zx_poly.global_parities.inverse(), zx_poly.n_qubits,
                         zx_poly.architecture)
    leg_chache = {}
    optimize_placement_matrix(zx_region.matrix, zx_region.angles)
    left, zx_region, right = optimize_region(left, zx_region, right, leg_chache, optimize_gauss=gaussian_step)

    left_ = ParityRegion(Mat2.id(zx_poly.n_qubits), zx_poly.n_qubits, zx_poly.architecture)
    right_ = ParityRegion(Mat2.id(zx_poly.n_qubits), zx_poly.n_qubits, zx_poly.architecture)
    region = synth_divide_conquer_step(left_, zx_region, right_, leg_chache, gaussian=gaussian_step)
    return left, region, right


def get_cx_count(left, region, right):
    circ_region = region_to_qiskit(region)

    cx_block = region_to_qiskit([left, right])
    return get_count(circ_region, type="two_qubit"), get_count(cx_block, type="two_qubit")
