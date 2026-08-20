"""
main.py
-------
Runs the full CS601T Assignment 1 pipeline for Group 23:

  Classification:
    - Linearly Separable (3 classes), logistic activation
    - Linearly Separable (3 classes), tanh activation
    - Non-linearly Separable (2 or 3 classes), logistic activation
    - Non-linearly Separable (2 or 3 classes), tanh activation

  Regression:
    - Univariate, linear activation
    - Bivariate, linear activation

Usage
-----
    python main.py                 # run everything, save all results
    python main.py --check-data    # only load & print dataset shapes
                                    # (use this FIRST once your real
                                    # data files are in place, to make
                                    # sure preprocessing.py parsed them
                                    # correctly before a full run)

All plots / metrics text files are written under results/.
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from src.utils import set_seed, train_test_split_per_class, train_test_split_simple
from src.perceptron import Perceptron
from src.one_vs_one import OneVsOnePerceptron
from src.metrics import classification_report, print_classification_report, rmse, percent_rmse
from src.visualize import (
    plot_error_vs_epoch, plot_multi_error_vs_epoch,
    plot_decision_region_pairwise, plot_decision_region_combined,
    plot_regression_fit_1d, plot_regression_fit_2d, plot_scatter_target_vs_pred,
)
import src.preprocessing as pp

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data", "Group23")
RESULTS_DIR = os.path.join(BASE, "results")

LS_DIR = os.path.join(DATA_DIR, "classification", "LS_Group23")
NLS_PATH = os.path.join(DATA_DIR, "classification", "NLS_Group23.txt")
UNIVARIATE_PATH = os.path.join(DATA_DIR, "regression", "UnivariateData", "23.csv")
BIVARIATE_PATH = os.path.join(DATA_DIR, "regression", "BivariateData", "23.csv")

LR = 0.01
EPOCHS = 500
SEED = 42


def ensure_dirs():
    for sub in ["classification/LS", "classification/NLS", "regression/Univariate",
                "regression/Bivariate", "comparison"]:
        os.makedirs(os.path.join(RESULTS_DIR, sub), exist_ok=True)


# --------------------------------------------------------------------
#                          CLASSIFICATION
# --------------------------------------------------------------------

def run_classification(X, y, dataset_name, out_subdir):
    """
    Runs the full classification pipeline (both activations) for one
    dataset (LS or NLS): split -> train OvO ensemble -> plots -> metrics.
    """
    classes = sorted(np.unique(y))
    X_train, y_train, X_test, y_test = train_test_split_per_class(
        X, y, train_frac=0.7, seed=SEED)

    print(f"\n[{dataset_name}] classes={classes} "
          f"train={X_train.shape[0]} test={X_test.shape[0]}")

    summary = {}

    for activation in ["logistic", "tanh"]:
        print(f"  -- activation: {activation} --")
        model = OneVsOnePerceptron(classes=classes, activation=activation,
                                    lr=LR, epochs=EPOCHS, seed=SEED)
        model.fit(X_train, y_train)

        act_dir = os.path.join(RESULTS_DIR, out_subdir, activation)
        os.makedirs(act_dir, exist_ok=True)

        # 1) error vs epoch -- one curve per pairwise classifier
        histories = {f"C{ci} vs C{cj}": h for (ci, cj), h in model.error_histories().items()}
        plot_multi_error_vs_epoch(
            histories, f"{dataset_name} ({activation}): avg error vs epoch",
            os.path.join(act_dir, "error_vs_epoch.png"))

        # 2) decision regions: pairwise + combined
        for (ci, cj), pw_model in model.pairwise_models.items():
            predict_fn = lambda Xg, ci=ci, cj=cj: model.predict_pair(ci, cj, Xg)
            plot_decision_region_pairwise(
                X_train, y_train, ci, cj, predict_fn,
                f"{dataset_name} ({activation}): decision region C{ci} vs C{cj}",
                os.path.join(act_dir, f"decision_region_C{ci}_vs_C{cj}.png"))

        plot_decision_region_combined(
            X_train, y_train, classes, model.predict,
            f"{dataset_name} ({activation}): combined decision region",
            os.path.join(act_dir, "decision_region_combined.png"))

        # 3) confusion matrix + accuracy/precision/recall/F-measure on TEST data
        y_pred_test = model.predict(X_test)
        report = classification_report(y_test, y_pred_test, classes)
        print_classification_report(report, title=f"{dataset_name} ({activation}), test data")

        with open(os.path.join(act_dir, "metrics.json"), "w") as f:
            json.dump({
                "accuracy": float(report["accuracy"]),
                "precision_per_class": dict(zip(map(str, classes), report["precision"].tolist())),
                "recall_per_class": dict(zip(map(str, classes), report["recall"].tolist())),
                "f_measure_per_class": dict(zip(map(str, classes), report["f_measure"].tolist())),
                "mean_precision": float(report["mean_precision"]),
                "mean_recall": float(report["mean_recall"]),
                "mean_f_measure": float(report["mean_f_measure"]),
                "confusion_matrix": report["confusion_matrix"].tolist(),
                "classes": [int(c) for c in classes],
            }, f, indent=2)

        summary[activation] = float(report["accuracy"])

    return summary


# --------------------------------------------------------------------
#                             REGRESSION
# --------------------------------------------------------------------

def run_regression(X, y, dataset_name, out_subdir, is_2d):
    X_train, y_train, X_test, y_test = train_test_split_simple(
        X, y, train_frac=0.7, seed=SEED)

    print(f"\n[{dataset_name}] train={X_train.shape[0]} test={X_test.shape[0]}")

    model = Perceptron(n_inputs=X.shape[1], activation="linear",
                        lr=LR, epochs=EPOCHS, seed=SEED)
    model.fit(X_train, y_train)

    out_dir = os.path.join(RESULTS_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # 1) error vs epoch
    plot_error_vs_epoch(model.error_history, f"{dataset_name}: avg error vs epoch",
                         os.path.join(out_dir, "error_vs_epoch.png"))

    # 2) RMSE / %RMSE on train and test
    pred_train = model.predict_raw(X_train)
    pred_test = model.predict_raw(X_test)
    metrics_out = {
        "rmse_train": rmse(y_train, pred_train),
        "rmse_test": rmse(y_test, pred_test),
        "percent_rmse_train": percent_rmse(y_train, pred_train),
        "percent_rmse_test": percent_rmse(y_test, pred_test),
    }
    print(f"  RMSE train={metrics_out['rmse_train']:.4f} "
          f"({metrics_out['percent_rmse_train']:.2f}%)  "
          f"test={metrics_out['rmse_test']:.4f} ({metrics_out['percent_rmse_test']:.2f}%)")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # 3) model output vs target
    if is_2d:
        plot_regression_fit_2d(X_train, y_train, pred_train,
                                f"{dataset_name}: train fit", os.path.join(out_dir, "fit_train.png"))
        plot_regression_fit_2d(X_test, y_test, pred_test,
                                f"{dataset_name}: test fit", os.path.join(out_dir, "fit_test.png"))
    else:
        plot_regression_fit_1d(X_train, y_train, pred_train,
                                f"{dataset_name}: train fit", os.path.join(out_dir, "fit_train.png"))
        plot_regression_fit_1d(X_test, y_test, pred_test,
                                f"{dataset_name}: test fit", os.path.join(out_dir, "fit_test.png"))

    # 4) scatter target vs predicted
    plot_scatter_target_vs_pred(y_train, pred_train, f"{dataset_name}: train (target vs model)",
                                 os.path.join(out_dir, "scatter_train.png"))
    plot_scatter_target_vs_pred(y_test, pred_test, f"{dataset_name}: test (target vs model)",
                                 os.path.join(out_dir, "scatter_test.png"))

    return metrics_out


# --------------------------------------------------------------------

def check_data():
    print("Checking dataset loading / shapes only (no training)...\n")
    try:
        X, y = pp.load_ls_data(LS_DIR)
        print(f"LS: X={X.shape} y={y.shape} classes={sorted(np.unique(y))}")
    except Exception as e:
        print(f"LS: FAILED to load ({e})")

    try:
        X, y = pp.load_nls_data(NLS_PATH, class_labels=(1, 2, 3))
        print(f"NLS: X={X.shape} y={y.shape} classes={sorted(np.unique(y))}")
    except Exception as e:
        print(f"NLS: FAILED to load ({e})")

    try:
        X, y = pp.load_univariate_regression(UNIVARIATE_PATH)
        print(f"Univariate: X={X.shape} y={y.shape}")
    except Exception as e:
        print(f"Univariate: FAILED to load ({e})")

    try:
        X, y = pp.load_bivariate_regression(BIVARIATE_PATH)
        print(f"Bivariate: X={X.shape} y={y.shape}")
    except Exception as e:
        print(f"Bivariate: FAILED to load ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-data", action="store_true",
                         help="Only load and print dataset shapes, then exit.")
    args = parser.parse_args()

    set_seed(SEED)
    ensure_dirs()

    if args.check_data:
        check_data()
        return

    overall_summary = {}

    # ---- Classification: LS ----
    X_ls, y_ls = pp.load_ls_data(LS_DIR)
    overall_summary["LS"] = run_classification(X_ls, y_ls, "LS", "classification/LS")

    # ---- Classification: NLS ----
    # NOTE: adjust class_labels=(1,2) to (1,2,3) if your NLS file has 3 classes
    X_nls, y_nls = pp.load_nls_data(NLS_PATH, class_labels=(1, 2))
    overall_summary["NLS"] = run_classification(X_nls, y_nls, "NLS", "classification/NLS")

    # ---- Regression: Univariate ----
    X_uni, y_uni = pp.load_univariate_regression(UNIVARIATE_PATH)
    overall_summary["Univariate"] = run_regression(
        X_uni, y_uni, "Univariate", "regression/Univariate", is_2d=False)

    # ---- Regression: Bivariate ----
    X_biv, y_biv = pp.load_bivariate_regression(BIVARIATE_PATH)
    overall_summary["Bivariate"] = run_regression(
        X_biv, y_biv, "Bivariate", "regression/Bivariate", is_2d=True)

    with open(os.path.join(RESULTS_DIR, "comparison", "overall_summary.json"), "w") as f:
        json.dump(overall_summary, f, indent=2, default=str)

    print("\nAll done. Results saved under results/.")


if __name__ == "__main__":
    main()
