import argparse
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

    metrics_all_models = []

    for model_name, (model_type, trainer_type) in MODEL_CHOICE.items():
        model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
        trainer = trainer_type(**model_params)
        trainer.cross_validation(all_ivectors, all_labels, k=k, verbose=True)
        metrics_all_models.append(trainer.cv_metrics)

    best_model = compare_metrics(metrics_all_models)

    return best_model


def compare_metrics(metrics_all_models: List[List[Tuple[str, Metrics]]]):
    # Todo: Put the metrics of all CVs from all models in here and compare.
    pass


def train_single_model(
    model_name: str,
    train_ivectors: ndarray,
    train_labels: ndarray,
    test_ivectors: ndarray,
    test_labels: ndarray,
    params: Dict,
    verbose: bool = False,
) -> Union[LogisticRegression, SVC]:

    trainer_type = MODEL_CHOICE[model_name][1]
    model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
    trainer = trainer_type(**model_params)

    model = trainer.train(
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

    trainer_type = MODEL_CHOICE[model_name][1]
    model_params = {key: params[key] for key in PARAM_CHOICE[model_name]}
    trainer = trainer_type(**model_params)

    final_model = trainer.train_final_model(
        ivectors,
        labels,
        verbose=verbose,
    )

    return final_model


def save_model(filename: str):
    # TODO: Use pickle to store models in a file.
    pass


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
    parser.add_argument("-v", "--verbose", action="store_true", help="Print training and testing metrics.")
    parser.add_argument("-k", "--keep_model", action="store_true", help="Store the chosen model in a file.")

    # If cross validation is run, the original split cannot be used (hence
    # the mututally exclusive group).
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--cross_val",
        action="store_true",
        help="Select a model with cross_validation.",
    )
    group.add_argument(
        "-o",
        "--original_split",
        action="store_true",
        help="Use the original train-test split given by VarDial.",
    )

    args = parser.parse_args()

    train_vec_file = "data/train.vec"
    train_txt_file = "data/train.txt"
    test_vec_file = "data/dev.vec"
    test_txt_file = "data/dev.txt"

    # TODO: The configuration parameters should be provided by a configuration
    # file in the future.
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

        train_ivectors = load_ivectors(train_vec_file)
        train_labels = load_labels(train_txt_file)
        test_ivectors = load_ivectors(test_vec_file)
        test_labels = load_labels(test_txt_file)

        if args.single_model:
            model = train_single_model(
                args.single_model,
                train_ivectors,
                train_labels,
                test_ivectors,
                test_labels,
                params,
                verbose=args.verbose
            )
        else:
            print(
                "You cannot train a final model or do cross validation with"
                "the original split."
            )

    else:
        all_ivectors, all_labels = combine_data(
            train_vec_file, train_txt_file, test_vec_file, test_txt_file
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
            model = train_single_model(
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
    train_time = time.strftime("%a-%d-%b-%Y", time.localtime())
    if args.keep_model:
        for m in (model, best_model, final_model):
            if m:
                # TODO: Determine how to call the filename.
                filename = train_time + model.__class__.__name__
                save_model(m, filename)
