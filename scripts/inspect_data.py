import matplotlib.pyplot as plt
import numpy as np

from dataset import load_ivectors
from dataset import load_labels


def feature_mean_sd_by_dialect():
    train_vec_file = "data/train.vec"
    train_txt_file = "data/train.txt"

    train_ivectors = load_ivectors(train_vec_file)
    train_labels = load_labels(train_txt_file)

    # Put the data for the dialects in 4 buckets.
    data_per_dialect = {0: [], 1: [], 2: [], 3: []}
    for ivector, dialect in zip(train_ivectors, train_labels):
        data_per_dialect[dialect].append(list(ivector))

    # Convert the lists set up above into np.arrays
    for dialect in data_per_dialect:
        data_per_dialect[dialect] = np.array(data_per_dialect[dialect])

    # Calculate the means of all features separately for all 4 dialects.
    means_by_dialect = {}
    for dialect in data_per_dialect:
        means_by_dialect[dialect] = data_per_dialect[dialect].mean(axis=0)
    # Put the means into one 2d dataframe.
    one, two, three, four = means_by_dialect.values()
    all_means = np.concatenate(([one], [two], [three], [four]), axis=0)

    # Calculate the standard deviations of all features separately for all
    # 4 dialects.
    sd_by_dialect = {}
    for dialect in data_per_dialect:
        sd_by_dialect[dialect] = data_per_dialect[dialect].std(axis=0)
    # Put the standard deviations into one 2d dataframe.
    one, two, three, four = sd_by_dialect.values()
    all_sds = np.concatenate(([one], [two], [three], [four]), axis=0)

    return all_means, all_sds


def identify_useless_features(all_means, all_sds):

    mean_spreads = []
    mean_sd = []

    for feat_num in range(all_means.shape[1]):
        feature_means = all_means[:, feat_num]
        mean_spreads.append(np.amax(feature_means) - np.amin(feature_means))
        feature_sd = all_sds[:, feat_num]
        mean_sd.append(np.mean(feature_sd))

    # If the distance between the maximum and minimum mean of the classes
    # is bigger than half the mean standard deviation of the four classes,
    # the feature is said to be useful.
    weighted_mean_spreads = np.array(mean_spreads) / (np.array(mean_sd) * 0.5)
    # print(weighted_mean_spreads)

    useless_features = np.argwhere(weighted_mean_spreads < 1)

    useless_features = list(useless_features.flatten())

    return useless_features


def plot_single_feature(column: int, train_ivectors, train_labels):
    """
    Plot a single feature by dialect.
    This makes it possible to identify whether different dialects
    behave differently and to see whether there are speaker clusters
    inside the dialects.

    4 plots (one per dialect) will be plotted.

    :param column: choose the feature to print per feature
    """
    column_by_dialect = {0: [], 1: [], 2: [], 3: []}

    for value, dialect in zip(train_ivectors[:, column], train_labels):
        column_by_dialect[dialect].append(value)

    column_by_dialect = {dialect: np.array(values) for dialect, values in column_by_dialect.items()}

    # DIALECT_MAP = {"LU": 0, "BE": 1, "ZH": 2, "BS": 3}
    fig, axs = plt.subplots(4)
    axs[0].title.set_text("Lucerne")
    axs[1].title.set_text("Bern")
    axs[2].title.set_text("Zurich")
    axs[3].title.set_text("Basel")
    for ax in axs:
        ax.set_xlabel('feature value', loc="right")
        ax.set_ylabel('no. utterance', loc="top")

    fig.suptitle("Feature " + str(column + 1))
    fig.subplots_adjust(hspace=0.7)

    for plot_num in range(4):
        dialect_values = column_by_dialect[plot_num]
        axs[plot_num].scatter(dialect_values, np.array(range(len(dialect_values))))

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
