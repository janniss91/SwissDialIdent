import matplotlib.pyplot as plt
import numpy as np

from dataset import load_ivectors
from dataset import load_labels

train_vec_file = "data/train.vec"
train_txt_file = "data/train.txt"

train_ivectors = load_ivectors(train_vec_file)
train_labels = load_labels(train_txt_file)


def feature_mean_sd_by_dialect():
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



def plot_single_feature(column: int):
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

    fig, axs = plt.subplots(4)
    fig.suptitle("Feature " + str(column + 1))

    for plot_num in range(4):
        dialect_values = column_by_dialect[plot_num]
        axs[plot_num].scatter(dialect_values, np.array(range(len(dialect_values))))

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == "__main__":

    all_means, all_sds = feature_mean_sd_by_dialect()

    print(identify_useless_features(all_means, all_sds))

    # for num in range(20):
    #     plot_single_feature(num)

# Not: 1
# Value 2 has different means
# 3 dfferent means
# 4
# 5 has lots of outliers to the right for class 2 (in range 0 to 3) -> Zürich, one speaker very different?
# 6 has very different variances
# 7 dialect 0 (Lucerne) has very different
