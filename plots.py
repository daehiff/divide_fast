import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RANDOM_PATHS = ["evaluations/data/random_0_1.csv"]
# "evaluations/data/random_0_2.csv",
# "evaluations/data/random_2_4.csv",
# "evaluations/data/random_4_6.csv",
# "evaluations/data/random_6_8.csv",
# "evaluations/data/random_8_10.csv",
# "evaluations/data/random_10_12.csv",
# "evaluations/data/random_12_14.csv",
# "evaluations/data/random_14_16.csv",
# "evaluations/data/random_16_18.csv",
# "evaluations/data/random_18_20.csv"]

GAUSS_FAST_PATHS = [
    "evaluations/data/gauss_fast_0_20.csv"
]

PYZX_FAST_PATH = [
    "evaluations/data/gauss_fast_0_1.csv"
]

MAXCUT_PATHS = ["evaluations/data/maxcut_0_20.csv"]

LIMITED_PATHS = ["evaluations/data/limited_random_0_20.csv"]

TWO_QUBIT_REDUCTION = "Two Qubit Reduction"
ONE_QUBIT_REDUCTION = "One Qubit Reduction"
CNOT_COUNT = "CNOT Count"
ONE_QUBIT_COUNT = "One Qubit Count"
N_QUBITS = "Number Qubits"
NR_REPITIONS = "Nr. Repetition"
NON_MATCHING_CX = "Non matching CNOTs"
TIME_DIFFERENCE = "time difference"
ARCH = "Architecture"
ALGORITHM = "Algorithm"
P_EDGE = "Edge propability"
CIRCUIT_REPS = "Circuit Repetitions"
N_PGS = "Number phase gadgets"
MAX_LEGS = "Maximum Legs"

MAX_CUT_COL_NAMES = {'two_qubit_reduction': TWO_QUBIT_REDUCTION,
                     'one_qubit_reduction': ONE_QUBIT_REDUCTION,
                     'cx_count': CNOT_COUNT,
                     'one_qubit_count': ONE_QUBIT_COUNT,
                     'n_qubits': N_QUBITS,
                     'n_rep': NR_REPITIONS,
                     'non_matching_cx': NON_MATCHING_CX,
                     'time_diff': TIME_DIFFERENCE,
                     'arch': ARCH,
                     'algorithm': ALGORITHM,
                     'p_edge': P_EDGE,
                     'circ_reps': CIRCUIT_REPS,
                     }

RANDOM_COL_NAMES = {'two_qubit_reduction': TWO_QUBIT_REDUCTION,
                    'one_qubit_reduction': ONE_QUBIT_REDUCTION,
                    'cx_count': CNOT_COUNT,
                    'one_qubit_count': ONE_QUBIT_COUNT,
                    'n_qubits': N_QUBITS,
                    'n_rep': NR_REPITIONS,
                    'non_matching_cx': NON_MATCHING_CX,
                    'time_diff': TIME_DIFFERENCE,
                    'arch': ARCH,
                    'algorithm': ALGORITHM,
                    'n_pgs': N_PGS,
                    'max_legs': MAX_LEGS,
                    }


def get_df(df_paths):
    df = pd.DataFrame()
    for path in df_paths:
        df_ = pd.read_csv(path)
        df = df.append(df_, ignore_index=True)
    return df


def plot_reproduce(show=False):
    df = get_df(LIMITED_PATHS)

    df = df[df.n_qubits == 9]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)
    ax = sns.barplot(data=df, x=N_PGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM, estimator=np.mean)
    plt.title("9 qubits")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/reproduce/9_qubit.pdf")
        plt.clf()
        plt.close()
    # plt.show()

    df = get_df(LIMITED_PATHS)

    df = df[df.n_qubits == 16]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)
    ax = sns.barplot(data=df, x=N_PGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM, estimator=np.mean)
    plt.title("16 qubits")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/reproduce/16_qubit.pdf")
        plt.clf()
        plt.close()

    df = get_df(LIMITED_PATHS)
    df = df[df.n_qubits == 25]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)
    ax = sns.barplot(data=df, x=N_PGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM, estimator=np.mean)
    plt.title("25 qubits")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/reproduce/25_qubit.pdf")
        plt.clf()
        plt.close()

    df = get_df(LIMITED_PATHS)
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)
    ax = sns.barplot(data=df, x=N_QUBITS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM, estimator=np.mean)
    plt.title("Scaling by Qubits")
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/reproduce/scaling_by_qubits.pdf")
        plt.clf()
        plt.close()


def plot_maxcut(show=False):
    df = get_df(MAXCUT_PATHS)
    df = df[df.algorithm != "original"]
    # df = df[df.algorithm != "tket"]
    # df = df[df.algorithm != "pyzx"]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df["one_qubit_reduction"] = df["one_qubit_reduction"] * 100.0
    df.rename(columns=MAX_CUT_COL_NAMES, inplace=True)
    print(df)

    ax = sns.barplot(data=df, x=N_QUBITS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/maxcut/two_qubit_nqubits.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=CIRCUIT_REPS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/maxcut/two_qubit_circuit_reps.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=P_EDGE, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/maxcut/two_qubit_pedge.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=ARCH, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/maxcut/two_qubit_architecture.pdf")
        plt.clf()
        plt.close()


def plot_random_experiment(show=False, loc=3):
    df = get_df(RANDOM_PATHS)
    # df = df[df.arch == "fully_connected"]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df["one_qubit_reduction"] = df["one_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)

    # df = df[df[ALGORITHM] != "pyzx"]
    # df = df[df[ALGORITHM] != "tket"]
    print(df)

    ax = sns.barplot(data=df, x=N_PGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/random/two_qubit_npgs.pdf")
        plt.clf()
        plt.close()
    ax = sns.barplot(data=df, x=N_QUBITS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/random/two_qubit_nqubits.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=ARCH, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/random/two_qubit_arch.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=MAX_LEGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/random/two_qubit_maxlegs.pdf")
        plt.clf()
        plt.close()


def plot_divide_fast_gauss_comparison(show=False, loc=3):
    df = get_df(GAUSS_FAST_PATHS)
    # df = df[df.arch == "fully_connected"]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df["one_qubit_reduction"] = df["one_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)

    ax = sns.barplot(data=df, x=N_QUBITS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/gauss_fast/two_qubit_reduction.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=N_QUBITS, y=TIME_DIFFERENCE, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/gauss_fast/time_difference_qubits.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=N_PGS, y=TIME_DIFFERENCE, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/gauss_fast/time_difference_npgs.pdf")
        plt.clf()
        plt.close()


def plot_pyzx_fast(show=False):
    df = get_df(PYZX_FAST_PATH)
    # df = df[df.arch == "fully_connected"]
    df["two_qubit_reduction"] = df["two_qubit_reduction"] * 100.0
    df["one_qubit_reduction"] = df["one_qubit_reduction"] * 100.0
    df.rename(columns=RANDOM_COL_NAMES, inplace=True)

    ax = sns.barplot(data=df, x=N_QUBITS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/pyzx_fast/two_qubit_reduction.pdf")
        plt.clf()
        plt.close()

    ax = sns.barplot(data=df, x=N_PGS, y=TWO_QUBIT_REDUCTION, hue=ALGORITHM)
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=4, title=None)
    plt.grid(True)
    if show:
        plt.show()
    else:
        plt.savefig("evaluations/plots/pyzx_fast/two_qubit_reduction_pgs.pdf")
        plt.clf()
        plt.close()


def main(show=False):
    plot_dirs = ["evaluations/plots/pyzx_fast", "evaluations/plots/reproduce",
                 "evaluations/plots/maxcut", "evaluations/plots/random",
                 "evaluations/plots/gauss_fast"]
    for dir in plot_dirs:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    plot_reproduce(show)
    plot_maxcut(show)
    plot_random_experiment(show=show)
    plot_divide_fast_gauss_comparison(show)
    plot_pyzx_fast(show)


if __name__ == '__main__':
    main()
