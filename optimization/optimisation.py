import json
import pickle
from fractions import Fraction

import numpy as np
import pytket
from pytket.utils import gen_term_sequence_circuit

from ZXPhasePoly import ZXPhasePoly
from synth.utils import is_able_to_swap, get_matrix, get_phases
from utils import generate_random_phase_poly, verify_equality_, clamp, tket_to_qiskit


def find_matching_parities_right(matrix, remaining_columns, column):
    par_column = matrix[:, column]
    for col in remaining_columns[remaining_columns.index(column) + 1:]:
        if np.all(par_column == matrix[:, col]):
            return col
    return None


def find_matching_parities_left(matrix, remaining_columns, column):
    par_column = matrix[:, column]
    for col in reversed(remaining_columns[:remaining_columns.index(column)]):
        if np.all(par_column == matrix[:, col]):
            return col
    return None


def is_able_to_swap_consider_phase(matrix, phases, next_col, column, pi_phases):
    if is_able_to_swap(matrix, next_col, column):
        return True
    elif phases[next_col] == 1:
        pi_phases.append(column)
        return True
    elif phases[column] == 1:
        pi_phases.append(next_col)
        return True
    else:
        type_i = "z" if 2 in matrix[:, next_col] else "x"
        type_j = "z" if 2 in matrix[:, column] else "x"
        assert type_j != type_i
        return False


def propagate_phase_gadgets(matrix, phases, remaining_columns):
    for column in remaining_columns:
        match = find_matching_parities_right(matrix, remaining_columns, column)
        if match is None:
            continue
        remaining_columns_ = [col for col in remaining_columns[remaining_columns.index(column):]]
        next_col = remaining_columns_.pop(0)
        pi_phases = []
        while remaining_columns_ and is_able_to_swap_consider_phase(matrix, phases, next_col, column, pi_phases):
            next_next_col = remaining_columns_.pop(0)
            if next_next_col == match:
                for col in pi_phases:
                    phases[col] = (phases[col] * -1)

                remaining_columns.remove(column)
                phases[match] = clamp(phases[match] + phases[column])
                phases[column] = 0.0
                return False
            else:
                next_col = next_next_col
    for column in reversed(remaining_columns):
        match = find_matching_parities_left(matrix, remaining_columns, column)
        if match is None:
            continue
        remaining_columns_ = [col for col in (remaining_columns[:remaining_columns.index(column)])]
        next_col = remaining_columns_.pop(-1)
        pi_phases = []
        while remaining_columns_ and is_able_to_swap_consider_phase(matrix, phases, next_col, column, pi_phases):
            next_next_col = remaining_columns_.pop(-1)
            if next_next_col == match:
                for col in pi_phases:
                    phases[col] = (phases[col] * -1)
                remaining_columns.remove(column)
                phases[match] = clamp(phases[match] + phases[column])
                phases[column] = Fraction(0, 1)
                return False
            else:
                next_col = next_next_col

    return True


def remove_collapsed_phase_gadgets(phases, remaining_columns):
    to_remove = []
    for col in remaining_columns:
        if phases[col] == 2 or phases[col] == 0:
            to_remove.append(col)
    return list(filter(lambda x: x not in to_remove, remaining_columns))


def optimize_zx_poly(poly: ZXPhasePoly) -> ZXPhasePoly:
    matrix = get_matrix(poly)
    phases = get_phases(poly)
    remaining_columns = list(range(matrix.shape[1]))

    converged = False
    while not converged:
        remaining_columns = remove_collapsed_phase_gadgets(phases, remaining_columns)
        converged = propagate_phase_gadgets(matrix, phases, remaining_columns)
    matrix = matrix[:, remaining_columns]
    phases = phases[remaining_columns]
    return ZXPhasePoly.from_matrix(matrix, phases, list(range(matrix.shape[1])),
                                   poly.architecture,
                                   poly.n_qubits,
                                   poly.global_parities)


def main(n_qubits=3):
    poly = generate_random_phase_poly(n_qubits, 100)
    poly_out = optimize_zx_poly(poly)
    assert (verify_equality_((poly.to_tket()), poly_out.to_tket()))


if __name__ == '__main__':
    for _ in range(1000):
        print(_)
        main()
