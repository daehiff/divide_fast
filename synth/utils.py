import numba
import numpy as np

from ZXPhasePoly import ZXPhasePoly


def update_matrix(control_idx: int, target_idx: int, matrix: np.array, cols_to_use: list):
    for i in cols_to_use:
        control = matrix[control_idx][i]
        target = matrix[target_idx][i]
        new_control = update_control(control, target)
        new_target = update_target(control, target)
        matrix[control_idx][i] = new_control
        matrix[target_idx][i] = new_target


def update_control(control: int, target: int) -> int:
    if target == 2:
        if control == 2:
            return 0
        elif control == 0:
            return 2
        else:
            raise Exception("invalid state control may only be in state 0, 2")
    else:
        return control


@numba.njit()
def update_target(control: int, target: int) -> int:
    if control == 1:
        if target == 1:
            return 0
        elif target == 0:
            return 1
        else:
            raise Exception("invalid state target may only be in state 0,1")
    else:
        return target


def get_matrix(zx_poly: ZXPhasePoly):
    matrix = np.ones((zx_poly.n_qubits, len(zx_poly.zx_phases))) * -1.0
    for idx, poly in enumerate(zx_poly.zx_phases):
        col = np.asarray([0 for _ in range(zx_poly.n_qubits)])
        if "x" in poly:
            col = col + np.asarray([int(i) for i in poly["x"][0]])
        if "z" in poly:
            col = col + 2 * np.asarray([int(i) for i in poly["z"][0]])
        matrix[:, idx] = col
    assert -1 not in matrix, "Phase gadget column was not parsed"
    return matrix


def get_phases(zx_poly: ZXPhasePoly):
    return np.asarray([zx_poly.get_phase(i) for i in range(len(zx_poly.zx_phases))])


def is_able_to_propagate(matrix, col_i, to_start=True):
    if to_start:
        for col_j in range(col_i):
            if not is_able_to_swap(matrix, col_j, col_i):
                return False
        return True
    else:
        for col_j in range(col_i, matrix.shape[1]):
            if not is_able_to_swap(matrix, col_j, col_i):
                return False
            return True


def is_able_to_swap(matrix, row_i, row_j):
    type_i = "z" if 2 in matrix[:, row_i] else "x"
    type_j = "z" if 2 in matrix[:, row_j] else "x"
    if type_i == type_j:
        return True

    # Note here set the legs to the same value (1.0)
    # While the non legs are set to different values
    # This way == operator counts the number of shared legs
    par_j = np.where(matrix[:, row_j] == 0.0, 'A', 1.0)
    par_i = np.where(matrix[:, row_i] == 0.0, 'B', 1.0)
    equals = par_j == par_i
    if np.sum(equals) % 2 == 0:
        return True
    return False
