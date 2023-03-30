from collections import deque
from io import BytesIO

import cairosvg
import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from pauliopt.utils import SVGBuilder
from pyzx import Mat2
from qiskit import QuantumCircuit

from ZXPhasePoly import RemainingCNOTTracker, ZXPhasePoly
from architecture import Architecture
from steiner import rec_steiner_gauss
from synth.utils import update_matrix, get_phases, get_matrix


def draw_matrix(matrix, file_out, cell_width=10, padding=3, red="#FF8888", green="#CCFFCC", in_terminal=True):
    im_width = 3 * (matrix.shape[1]) * (cell_width + padding) + (cell_width + padding)
    img_height = (matrix.shape[0] + 1) * (cell_width + padding) + (cell_width + padding)
    builder = SVGBuilder(im_width, img_height)

    def get_coordinates(col, row):
        offset = int(padding / 2.0) + int(cell_width / 2.0)
        return offset + col * (cell_width + padding), offset + (row * (cell_width + padding))

    for row in range(matrix.shape[0]):
        col_coord, row_coord = get_coordinates(0, row + 2)
        builder.line((0, row_coord), (col_coord, row_coord))

    for col in range(matrix.shape[1]):
        # Building the head of the Phase gadget
        pre_phase_col_coord, pre_phase_row_coord = get_coordinates(3 * col + 1, 1)
        phase_col_coord, phase_row_coord = get_coordinates(3 * col + 2, 1)
        builder.line((pre_phase_col_coord, pre_phase_row_coord),
                     (phase_col_coord, phase_row_coord))
        for row in range(0, matrix.shape[0]):
            col_coord, row_coord = get_coordinates(3 * col, row + 2)

            # Line to next Column
            next_col, next_row = get_coordinates(3 * col + 3, row + 2)
            builder.line((col_coord, row_coord), (next_col, row_coord))

            # Leg of the Phasegadget
            color = red if matrix[row, col] == 1 else green
            if matrix[row, col] != 0:
                builder.line((col_coord, row_coord), (pre_phase_col_coord, pre_phase_row_coord))
                builder.circle((col_coord, row_coord), int(cell_width / 2.0), color)

        pre_color_top = red if 1 not in matrix[:, col] else green
        color_top = red if 1 in matrix[:, col] else green
        builder.circle((pre_phase_col_coord, pre_phase_row_coord), int(cell_width / 2.0), pre_color_top)
        builder.circle((phase_col_coord, phase_row_coord), int(cell_width / 2.0), color_top)

    if in_terminal:
        output = cairosvg.svg2png(repr(builder))
        pil_img = Image.open(BytesIO(output)).convert('RGBA')
        print(pil_img)
        plt.imshow(pil_img)
        plt.title(file_out)
        plt.show()
    else:
        with open(f"{file_out}.svg", "w") as f:
            f.write(repr(builder))


def decompose_phase_gadget(circ: QuantumCircuit, column: np.array, angle: float, arch) -> QuantumCircuit:
    cnot_ladder, q0 = find_minimal_cx_assignment(column, arch)
    if len(cnot_ladder) > 0:
        target, control = 0, 0
        for (control, target) in reversed(cnot_ladder):
            circ.cx(control, target)

        if 1 in column:
            circ.rx(angle * np.pi, q0)
        else:
            circ.rz(angle * np.pi, q0)

        for (control, target) in cnot_ladder:
            circ.cx(control, target)
    else:
        target = np.argmax(column)
        if 1 in column:
            circ.rx(angle * np.pi, target)
        else:
            circ.rz(angle * np.pi, target)
    return circ


def get_distance_map(architecture):
    return 4 * architecture.floyd_warshall_dist - 2


def decompose_cnot_ladder_z(ctrl: int, trg: int, arch: Architecture):
    cnot_ladder = []
    shortest_path = arch.fw_shortest_path(ctrl, trg)

    prev = ctrl
    for current in shortest_path[1:-1]:
        # cnot_ladder.append((prev, current))
        cnot_ladder.append((current, prev))
        cnot_ladder.append((prev, current))
        prev = current
    cnot_ladder.append((shortest_path[-2], trg))
    return reversed(cnot_ladder)


def decompose_cnot_ladder_x(fst: int, snd: int, arch: Architecture):
    cnot_ladder = []
    shortest_path = arch.fw_shortest_path(fst, snd)
    prev = fst
    for current in shortest_path[1:-1]:
        cnot_ladder.append((prev, current))
        cnot_ladder.append((current, prev))
        # cnot_ladder.append((prev, current))

        prev = current
    cnot_ladder.append((snd, shortest_path[-2]))
    return reversed(cnot_ladder)


def find_minimal_cx_assignment(column: np.array, arch: Architecture, q0=None):
    G = nx.Graph()
    for i in range(len(column)):
        G.add_node(i)

    for i in range(len(column)):
        for j in range(len(column)):
            if column[i] != 0 and column[j] != 0 and i != j:
                G.add_edge(i, j, weight=4 * arch.fw_dist(i, j) - 2)

    # Algorithm by Gogioso et. al. (https://arxiv.org/pdf/2206.11839.pdf) to find qubit assignment with MST
    mst_branches = list(nx.minimum_spanning_edges(G, data=False, algorithm="prim"))
    incident = {q: set() for q in range(len(column))}
    for fst, snd in mst_branches:
        incident[fst].add((fst, snd))
        incident[snd].add((snd, fst))

    q0 = np.argmax(column)  # Assume that 0 is always the first qubit aka first non zero
    visited = set()
    queue = deque([q0])
    cnot_ladder = []
    while queue:
        q = queue.popleft()
        visited.add(q)
        for tail, head in incident[q]:
            if head not in visited:
                if 1 in column:
                    cnot_ladder += decompose_cnot_ladder_x(head, tail, arch)
                else:
                    cnot_ladder += decompose_cnot_ladder_z(head, tail, arch)
                queue.append(head)
    return cnot_ladder, q0


def get_col_effect(col, arch, leg_chache):
    col_id = "".join([str(int(el)) for el in col])
    if col_id in leg_chache.keys():
        return leg_chache[col_id]
    else:
        cnot_amount = len(find_minimal_cx_assignment(col, arch)[0])
        leg_chache[col_id] = cnot_amount
    return cnot_amount


def _get_effect(i, j, matrix, columns_to_use, arch, leg_chache=None):
    if leg_chache is None:
        leg_chache = {}

    effect = 0
    for col in columns_to_use:
        effect -= 2 * get_col_effect(matrix[:, col], arch, leg_chache)

    matrix_ = matrix.copy()
    update_matrix(i, j, matrix_, columns_to_use)

    for col in columns_to_use:
        effect += 2 * get_col_effect(matrix_[:, col], arch, leg_chache)
    return effect


class ZXRegion:
    def __init__(self):
        pass

    def get_effect(self, direction):
        raise NotImplemented("get_effect is not implemented on this object")

    def to_qiskit(self):
        raise NotImplemented("to_qiskit is not implemented on this object")

    def update(self, ctrl, target, direction):
        raise NotImplemented("update is not implemented on this object")


class ZXPolyRegion(ZXRegion):
    def __init__(self, matrix, angles, n_qubits, architecture):
        super().__init__()
        self.matrix = matrix
        self.angles = angles
        self.n_qubits = n_qubits
        self.architecture = architecture

    def get_effect(self, **kwargs):
        effect = np.zeros((self.n_qubits, self.n_qubits))
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i != j:
                    effect[i, j] += _get_effect(i, j, self.matrix, list(range(self.matrix.shape[1])), self.architecture)
        return effect

    def to_qiskit(self):
        circ = QuantumCircuit(self.n_qubits)
        for col in range(self.matrix.shape[1]):
            circ = decompose_phase_gadget(circ, self.matrix[:, col], self.angles[col], self.architecture)
        return circ

    def update(self, ctrl, target, **kwargs):
        update_matrix(ctrl, target, self.matrix, list(range(self.matrix.shape[1])))

    def get_effect_local(self, ctrl, target, leg_chache=None):
        if leg_chache is None:
            leg_chache = {}
        return _get_effect(ctrl, target, self.matrix, range(self.matrix.shape[1]), self.architecture, leg_chache)

    @staticmethod
    def from_zx_polynomial(zx_poly: ZXPhasePoly):
        matrix = get_matrix(zx_poly)
        angels = get_phases(zx_poly)
        return ZXPolyRegion(matrix, angels, zx_poly.n_qubits, zx_poly.architecture)


class ParityRegion(ZXRegion):
    def __init__(self, mat: Mat2, n_qubits, architecture):
        super().__init__()
        self.mat = mat
        self.n_qubits = n_qubits
        self.architecture = architecture

    def get_effect_local_heuristic(self, ctrl_added, target_added, dist, direction):
        if direction == "r":
            for (ctrl, target) in reversed(self.added_cnots):
                if ctrl_added == ctrl and target_added == target:
                    return -dist[ctrl, target]
                elif ctrl_added == target or target_added == ctrl:
                    return dist[ctrl, target]
            return dist[ctrl_added, target_added]
        else:
            for (ctrl, target) in self.added_cnots:
                if ctrl_added == ctrl and target_added == target:
                    return -dist[ctrl, target]
                elif ctrl_added == target or target_added == ctrl:
                    return dist[ctrl, target]
            return dist[ctrl_added, target_added]

    def get_effect_local(self, ctrl, target, direction):
        mat_ = self.mat.copy()
        self.__update_mat(ctrl, target, mat_, direction)

        cnots_ = RemainingCNOTTracker()
        rec_steiner_gauss(mat_, self.architecture, y=cnots_, full_reduce=True)

        cnots = RemainingCNOTTracker()
        rec_steiner_gauss(self.mat.copy(), self.architecture, y=cnots, full_reduce=True)
        return len(cnots_.remaining_cnots) - len(cnots.remaining_cnots)

    def get_added_gates(self):
        cnots = RemainingCNOTTracker()
        rec_steiner_gauss(self.mat.copy(), self.architecture, full_reduce=True, y=cnots)
        return cnots.remaining_cnots

    def get_effect(self, direction):
        effect = np.zeros((self.n_qubits, self.n_qubits))
        for i in range(self.n_qubits):
            for j in range(self.n_qubits):
                if i != j:
                    mat_ = self.mat.copy()
                    self.__update_mat(i, j, mat_, direction)

                    mat = self.mat.copy()

                    effect[i, j] += len(mat_.to_cnots(optimize=True)) - len(mat.to_cnots(optimize=True))
        return effect

    def to_qiskit(self):
        circ = QuantumCircuit(self.n_qubits)
        cnots = RemainingCNOTTracker()
        rec_steiner_gauss(self.mat.copy(), self.architecture, y=cnots, full_reduce=True)
        for (control, target) in reversed(cnots.remaining_cnots):
            circ.cx(control, target)
        return circ

    def __update_mat(self, ctrl, trgt, mat, direction):
        if direction == "r":
            mat.row_add(ctrl, trgt)
        else:
            mat.col_add(trgt, ctrl)

    def update(self, ctrl, trgt, direction):
        self.__update_mat(ctrl, trgt, self.mat, direction)
