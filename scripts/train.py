import argparse
import numpy as np
import os
import pickle
import re
import time
from numpy import ndarray
from sklearn.svm import SVC
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from dataset import combine_data
from dataset import load_ivectors
from dataset import load_labels
from logistic_regression import LogisticRegression
from logistic_regression import LogisticRegressionTrainer
from metrics import Metrics
from svm import SVMTrainer

STORED_MODEL_DIR = "stored_models"

MODEL_CHOICE = {
    "LogisticRegression": (LogisticRegression, LogisticRegressionTrainer),
    "SVM": (SVC, SVMTrainer),
}

PARAM_CHOICE = {
    "LogisticRegression": ("n_epochs", "batch_size", "lr", "log_interval"),
    "SVM": ("c", "kernel", "degree", "max_iter"),
}


def select_model(
    ivectors: ndarray,
    labels: ndarray,
    params: Dict,
    k: int = 10,
    verbose: bool = False,
) -> Union[LogisticRegression, SVC]:
    """
    Select a best model from all model types with cross validation.

    :param ivectors: all shuffled i-vectors from train and test set
    :param labels: all shuffled (equivalent to i-vectors) labels from
    train and test set
    :param params: the dictionary of parameters for all models
    :param k: determines the number of splits of cross validation
    :param verbose: if true, losses and metrics are printed during training
    """

    all_models = []
    metrics_all_models = []

    for model_name, (model_type, trainer_type) in MODEL_CHOICE.items():
        model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
        trainer = trainer_type(**model_params)
        trainer.cross_validation(all_ivectors, all_labels, k=k, verbose=True)
        metrics_all_models.append(trainer.cv_metrics)
        all_models.append(trainer.cv_models)

    best_model = compare_metrics(all_models, metrics_all_models)

    return best_model


def compare_metrics(
    all_models: List[Union[LogisticRegression, SVC]],
    metrics_all_models: List[List[Tuple[str, Metrics]]],
):
    """
    Compare the metrics of the different models to choose the best model.

    At the moment only macro-averaged F1 is used as a metrics to determine
    the best model.
    A more sophisticated calculation could be inserted if necessary.

    :param all_models: all models that have been trained during cross validation
    :param metrics_all_models: the metrics models for all objects
    :return: the model with the best macro-averaged F1 score
    """
    best_model = None
    best_f1 = 0.0

    for model, metrics in zip(all_models, metrics_all_models):
        f1 = metrics.macroavg["f1-score"]
        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    return best_model


def train_single_model(
    model_name: str,
    train_ivectors: ndarray,
    train_labels: ndarray,
    test_ivectors: ndarray,
    test_labels: ndarray,
    params: Dict,
    verbose: bool = False,
) -> Union[LogisticRegression, SVC]:
    """
    Train a single model.

    :param model_name: type of the model to be trained
    :param train_ivectors: training i-vectors
    :param train_labels: training i-vectors
    :param test_ivectors: test i-vectors
    :param test_labels: test i-vectors
    :param params: the dictionary of parameters for all models
    :param verbose: if true, losses and metrics are printed during training
    """
    trainer_type = MODEL_CHOICE[model_name][1]
    model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
    trainer = trainer_type(**model_params)

    model, metrics = trainer.train(
        train_ivectors,
        train_labels,
        test_ivectors,
        test_labels,
        verbose=verbose,
    )

    return model


def train_final_model(
    model_name: str,
    ivectors: ndarray,
    labels: ndarray,
    params: Dict,
    verbose: bool = False,
) -> Union[LogisticRegression, SVC]:
    """
    Train a final model with the whole dataset.

    :param model_name: type of the model to be trained
    :param ivectors: all shuffled i-vectors from train and test set
    :param labels: all shuffled (equivalent to i-vectors) labels from
    train and test set
    :param params: the dictionary of parameters for all models
    :param verbose: if true, losses and metrics are printed during training
    """
    trainer_type = MODEL_CHOICE[model_name][1]
    model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
    trainer = trainer_type(**model_params)

    final_model = trainer.train_final_model(
        ivectors,
        labels,
        verbose=verbose,
    )

    return final_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("config_file", help="A configuration file that specifies training parameters.")
    parser.add_argument(
        "-s",
        "--single_model",
        choices=["LogisticRegression", "SVM"],
        help="Train a single model.",
    )
    parser.add_argument(
        "-f",
        "--final_model",
        choices=["LogisticRegression", "SVM"],
        help="Train a single model with the entire dataset",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print training and testing metrics.",
    )
    parser.add_argument(
        "-g",
        "--gan_ivec_file",
        type=str,
        help="Include artificially created GAN data from specified file.",
    )

    # If cross validation is run, the original split cannot be used (hence
    # the mututally exclusive group).
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--cross_val",
        action="store_true",
        help="Select a model with cross_validation.",
    )

    # TODO: Make original split the default.
    group.add_argument(
        "-o",
        "--original_split",
        action="store_true",
        help="Use the original train-test split given by VarDial.",
    )

    args = parser.parse_args()

    TRAIN_VEC_FILE = "data/train.vec"
    TRAIN_TXT_FILE = "data/train.txt"
    TEST_VEC_FILE = "data/dev.vec"
    TEST_TXT_FILE = "data/dev.txt"

    # TODO: The configuration parameters should be provided by a configuration
    # file in the future.
    # Typical values SVM: 0.0001 < gamma < 10; 0.1 < c < 100
    # kernel types: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
    params = {
        "n_epochs": 10,
        "batch_size": 10,
        "lr": 0.01,
        "log_interval": 50,
        "c": 1.0,
        "kernel": "rbf",
        "degree": 3,
        "max_iter": -1,
    }

    model = None
    best_model = None
    final_model = None

    if args.original_split:

        train_ivectors = load_ivectors(TRAIN_VEC_FILE)
        train_labels = load_labels(TRAIN_TXT_FILE)
        test_ivectors = load_ivectors(TEST_VEC_FILE)
        test_labels = load_labels(TEST_TXT_FILE)

        if args.gan_ivec_file:
            gan_ivec_file = args.gan_ivec_file
            split_fname = re.split("-|\.", gan_ivec_file)
            gan_txt_file = split_fname[0] + "-labels-" + split_fname[2] + ".txt"

            gan_ivectors = load_ivectors(args.gan_ivec_file)
            gan_labels = load_labels(gan_txt_file)

            train_ivectors = np.concatenate((train_ivectors, gan_ivectors), axis=0)
            train_labels = np.concatenate((train_labels, gan_labels), axis=0)

        if args.single_model:
            model = train_single_model(
                args.single_model,
                train_ivectors,
                train_labels,
                test_ivectors,
                test_labels,
                params,
                verbose=args.verbose,
            )
        else:
            print(
                "You cannot train a final model or do cross validation with"
                "the original split."
            )

    else:
        if args.gan_ivec_file:
            print("At the moment, you can use GAN-generated data only with single model training.")
            print("The GAN data will not be part of the training.")

        all_ivectors, all_labels = combine_data(
            TRAIN_VEC_FILE, TRAIN_TXT_FILE, TEST_VEC_FILE, TEST_TXT_FILE
        )

        if args.cross_val:
            best_model = select_model(
                # TODO: Make parameters configurable.
                all_ivectors,
                all_labels,
                params,
                k=3,
                verbose=args.verbose,
            )

        if args.single_model:

            # TODO: Introduce train_test_split functionality (set parameter somewhere).
            model, metrics = train_single_model(
                args.single_model,
            )

        if args.final_model:
            final_model = train_final_model(
                args.final_model,
                all_ivectors,
                all_labels,
                params,
                args.verbose,
            )

    # Store all models that have been set up if desired.
    store_time = time.strftime("%d-%b-%Y-%H:%M:%S", time.localtime())
    for denom, m in (("", model), ("best", best_model), ("final", final_model)):
        if m:
            model_name = m.__class__.__name__
            store = input(
                "\n\nDo you want to store the {} model ({}) [yes|no]: ".format(
                    denom, model_name
                )
            )

            if store == "yes":
                # TODO: Determine how to call the filename.
                filename = store_time + "-" + model_name + ".sav"
                print(filename)

                path_to_store = os.path.join(STORED_MODEL_DIR, filename)
                pickle.dump(m, open(path_to_store, 'wb'))
