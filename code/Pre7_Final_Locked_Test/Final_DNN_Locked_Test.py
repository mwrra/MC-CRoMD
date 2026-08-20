# -*- coding: utf-8 -*-
"""
Final_DNN_Locked_Test.py

Final locked-test evaluation for the revised MC-CRoMD DNN workflow.

SCIENTIFIC POLICY
-----------------
1) The three final DNN configurations are HARD-CODED below and were frozen
   BEFORE loading Test, based only on Train/Validation results.
2) This program does NOT search, rank, tune, or compare alternative patch sizes,
   feature methods, K values, architectures, hyperparameters, or random seeds.
3) The final seed is fixed a priori as 42. It is NOT chosen from Test.
4) The program loads the already-saved best validation checkpoint
   (Best_Model.keras) for the frozen configuration and seed 42.
5) No model fitting, scaler fitting, feature-selection fitting, threshold tuning,
   or early stopping is performed in this program.
6) Test is loaded only for the three frozen final evaluations.
7) All metrics are derived from the SAME per-sample prediction file.
8) Macro and weighted metrics are both saved explicitly.
9) Class-wise metrics and confusion-matrix counts are saved.
10) A group-aware bootstrap confidence interval is computed using SourceGroupID
    when available; otherwise sample-level bootstrap is used.
11) A checkpoint and lock files are written so an interrupted run can resume
    without changing any frozen decision.
12) Re-running after FINAL_TEST_COMPLETED.lock exists is refused by default.

FROZEN DNN CONFIGURATIONS
-------------------------
Binary      : patch 64  | Mutclass | K=3600 | seed=42
Three_Class : patch 256 | Variance | K=225  | seed=42
Five_Class  : patch 256 | Logistic | K=200  | seed=42

These were selected BEFORE Test using mean Validation Accuracy across seeds
42, 43, and 44, with mean Validation Macro-F1 as the secondary criterion.

Input DNN architecture/training policy was already fixed during validation:
Dense(128)-BN-Dropout(0.30)-Dense(64)-BN-Dropout(0.20)-Dense(32)
Adam, initial LR=0.001, batch size=64, max epochs=100,
early stopping on val_loss (patience=15), ReduceLROnPlateau,
He-normal initialization.

IMPORTANT
---------
Run this program only after you are satisfied that the frozen configurations
above are final. Do not change them after observing Test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# Frozen scientific decisions -- DO NOT MODIFY AFTER TEST IS OPENED
# =============================================================================

FINAL_SEED = 42
SOURCE_VARIANT = "StandardScaled"
TARGET_COLUMN = "label"

FROZEN_CONFIGS: dict[str, dict[str, Any]] = {
    "Binary": {
        "patch_size": 64,
        "feature_method": "Mutclass",
        "k": 3600,
        "seed": FINAL_SEED,
    },
    "Three_Class": {
        "patch_size": 256,
        "feature_method": "Variance",
        "k": 225,
        "seed": FINAL_SEED,
    },
    "Five_Class": {
        "patch_size": 256,
        "feature_method": "Logistic",
        "k": 200,
        "seed": FINAL_SEED,
    },
}

SCENARIO_ORDER = ("Binary", "Three_Class", "Five_Class")

DEFAULT_BASE_DIR = Path(r"G:\My Research About Lung Canser\INASS")
DEFAULT_OUTPUT_DIR_NAME = "7-DNNFinalLockedTest"
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260819

SAMPLE_ID_CANDIDATES = ("ImageSHA256", "FileName", "ImgName")
GROUP_ID_CANDIDATES = ("SourceGroupID", "LC25000_GroupID", "group_id")


# =============================================================================
# Utility helpers
# =============================================================================

def utc_safe_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def find_existing_file(folder: Path, stem: str) -> Path:
    candidates = (
        folder / f"{stem}.csv.gz",
        folder / f"{stem}.csv",
        folder / f"{stem}.xlsx",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of:\n" + "\n".join(str(p) for p in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def choose_first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def frozen_config_signature() -> str:
    payload = json.dumps(
        {
            "final_seed": FINAL_SEED,
            "source_variant": SOURCE_VARIANT,
            "configs": FROZEN_CONFIGS,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# =============================================================================
# Frozen-input discovery and validation
# =============================================================================

def paths_for_config(base_dir: Path, scenario: str) -> dict[str, Path]:
    cfg = FROZEN_CONFIGS[scenario]
    patch = int(cfg["patch_size"])
    method = str(cfg["feature_method"])
    k = int(cfg["k"])
    seed = int(cfg["seed"])

    preprocessing_dir = base_dir / "3-DataPreprocessing" / str(patch)
    feature_selection_dir = base_dir / "4-FeatureSelection" / str(patch)
    dnn_validation_dir = base_dir / "6-DNNValidation" / str(patch)

    selected_feature_path = (
        feature_selection_dir
        / scenario
        / method
        / f"Selected_Features_K{k:03d}.xlsx"
    )

    run_dir = (
        dnn_validation_dir
        / "Runs"
        / scenario
        / method
        / f"K{k:05d}"
        / f"Seed_{seed}"
    )

    test_dir = preprocessing_dir / scenario / SOURCE_VARIANT
    test_path = find_existing_file(test_dir, "Test")

    return {
        "preprocessing_dir": preprocessing_dir,
        "feature_selection_dir": feature_selection_dir,
        "dnn_validation_dir": dnn_validation_dir,
        "selected_feature_path": selected_feature_path,
        "run_dir": run_dir,
        "model_path": run_dir / "Best_Model.keras",
        "run_metadata_path": run_dir / "Run_Metadata.json",
        "validation_selected_features_path": run_dir / "Selected_Features.xlsx",
        "test_path": test_path,
    }


def read_selected_features(path: Path, expected_k: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path)
    if "Feature" not in frame.columns:
        raise KeyError(f"'Feature' column not found in {path}")
    features = frame["Feature"].dropna().astype(str).str.strip().tolist()
    features = [x for x in features if x]

    if len(features) != expected_k:
        raise ValueError(
            f"{path}: expected {expected_k} features but found {len(features)}."
        )
    if len(features) != len(set(features)):
        raise ValueError(f"Duplicate selected features found in {path}.")
    return features


def verify_against_validation_artifacts(
    paths: dict[str, Path],
    scenario: str,
    selected_features: list[str],
) -> dict[str, Any]:
    cfg = FROZEN_CONFIGS[scenario]

    for key in ("model_path", "run_metadata_path", "validation_selected_features_path"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing frozen validation artifact: {paths[key]}")

    run_metadata = json.loads(
        paths["run_metadata_path"].read_text(encoding="utf-8")
    )

    checks = {
        "scenario_match": str(run_metadata.get("scenario")) == scenario,
        "feature_method_match": str(run_metadata.get("feature_method")) == str(cfg["feature_method"]),
        "k_match": int(run_metadata.get("k")) == int(cfg["k"]),
        "seed_match": int(run_metadata.get("seed")) == int(cfg["seed"]),
        "source_variant_match": str(run_metadata.get("source_variant")) == SOURCE_VARIANT,
    }

    if not all(checks.values()):
        raise RuntimeError(
            f"{scenario}: frozen Run_Metadata.json does not match hard-coded final config: {checks}"
        )

    validation_features = read_selected_features(
        paths["validation_selected_features_path"],
        expected_k=int(cfg["k"]),
    )

    if validation_features != selected_features:
        raise RuntimeError(
            f"{scenario}: Pre5 selected-feature list differs from the list saved "
            "inside the frozen DNN validation run."
        )

    class_names = [str(x) for x in run_metadata.get("class_names", [])]
    if len(class_names) < 2:
        raise ValueError(f"{scenario}: invalid class_names in Run_Metadata.json")

    return {
        "run_metadata": run_metadata,
        "class_names": class_names,
        "validation_artifact_checks": checks,
    }


# =============================================================================
# Predictions and metrics
# =============================================================================

def predict_probabilities(
    model: keras.Model,
    X: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(model.predict(X, verbose=0), dtype=float)

    if n_classes == 2:
        positive = raw.reshape(-1)
        probabilities = np.column_stack([1.0 - positive, positive])
        predictions = (positive >= 0.5).astype(int)
    else:
        probabilities = raw
        predictions = np.argmax(probabilities, axis=1).astype(int)

    if probabilities.shape != (len(X), n_classes):
        raise ValueError(
            f"Probability matrix shape {probabilities.shape} does not match "
            f"expected {(len(X), n_classes)}."
        )

    return predictions, probabilities


def compute_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
) -> float:
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))

        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        return float(
            roc_auc_score(
                y_bin,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        return float("nan")


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_Macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Recall_Macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "F1_Macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "Precision_Weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "Recall_Weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "F1_Weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "ROC_AUC_OVR_Macro": compute_auc(
            y_true=y_true,
            probabilities=probabilities,
            n_classes=n_classes,
        ),
    }


def classwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "Class_Index": np.arange(len(class_names)),
            "Class_Name": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )


# =============================================================================
# Group-aware bootstrap confidence intervals
# =============================================================================

def bootstrap_distribution(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    n_classes: int,
    group_values: np.ndarray | None,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    if repeats <= 0:
        return pd.DataFrame(), "disabled"

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    if group_values is not None:
        groups = np.asarray(group_values).astype(str)
        unique_groups = np.unique(groups)
        group_to_indices = {
            group: np.flatnonzero(groups == group)
            for group in unique_groups
        }
        bootstrap_unit = "SourceGroupID"
    else:
        unique_groups = None
        group_to_indices = None
        bootstrap_unit = "sample"

    for iteration in range(1, repeats + 1):
        if unique_groups is not None:
            sampled_groups = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            sampled_indices = np.concatenate(
                [group_to_indices[group] for group in sampled_groups]
            )
        else:
            sampled_indices = rng.integers(
                low=0,
                high=len(y_true),
                size=len(y_true),
            )

        yt = y_true[sampled_indices]
        yp = y_pred[sampled_indices]
        pp = probabilities[sampled_indices]

        # Some bootstrap draws may omit one or more classes.
        # Accuracy remains defined; macro metrics are calculated using labels present
        # in the fixed class index space through sklearn's multiclass handling.
        metrics = calculate_all_metrics(
            yt,
            yp,
            pp,
            n_classes=n_classes,
        )

        rows.append(
            {
                "Iteration": iteration,
                **metrics,
            }
        )

    return pd.DataFrame(rows), bootstrap_unit


def confidence_interval_summary(
    bootstrap_frame: pd.DataFrame,
    point_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for metric_name, point_value in point_metrics.items():
        if bootstrap_frame.empty or metric_name not in bootstrap_frame.columns:
            low = high = float("nan")
        else:
            values = pd.to_numeric(
                bootstrap_frame[metric_name],
                errors="coerce",
            ).dropna()
            if values.empty:
                low = high = float("nan")
            else:
                low = float(np.quantile(values, 0.025))
                high = float(np.quantile(values, 0.975))

        rows.append(
            {
                "Metric": metric_name,
                "Point_Estimate": float(point_value),
                "CI_95_Lower": low,
                "CI_95_Upper": high,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Final evaluation for one scenario
# =============================================================================

def evaluate_one_scenario(
    *,
    base_dir: Path,
    output_dir: Path,
    scenario: str,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    cfg = FROZEN_CONFIGS[scenario]
    paths = paths_for_config(base_dir, scenario)

    scenario_output = output_dir / scenario
    scenario_output.mkdir(parents=True, exist_ok=True)

    selected_features = read_selected_features(
        paths["selected_feature_path"],
        expected_k=int(cfg["k"]),
    )

    validation_check = verify_against_validation_artifacts(
        paths=paths,
        scenario=scenario,
        selected_features=selected_features,
    )
    class_names = validation_check["class_names"]
    n_classes = len(class_names)

    test_frame = read_table(paths["test_path"])
    test_frame.columns = [str(c).strip() for c in test_frame.columns]

    if TARGET_COLUMN not in test_frame.columns:
        raise KeyError(f"{scenario}: Test is missing target column {TARGET_COLUMN!r}.")

    if "Split" in test_frame.columns:
        split_values = set(test_frame["Split"].astype(str).str.strip().unique())
        if split_values != {"Test"}:
            raise RuntimeError(
                f"{scenario}: expected only Split='Test', found {sorted(split_values)}"
            )

    missing_features = sorted(set(selected_features).difference(test_frame.columns))
    if missing_features:
        raise KeyError(
            f"{scenario}: Test is missing {len(missing_features)} selected features. "
            f"Examples: {missing_features[:10]}"
        )

    label_to_index = {label: i for i, label in enumerate(class_names)}
    labels_raw = test_frame[TARGET_COLUMN].astype(str).to_numpy()
    unknown_labels = sorted(set(labels_raw).difference(label_to_index))
    if unknown_labels:
        raise ValueError(
            f"{scenario}: Test contains labels absent from frozen model classes: {unknown_labels}"
        )

    y_true = np.asarray([label_to_index[x] for x in labels_raw], dtype=int)
    X_test = test_frame[selected_features].to_numpy(dtype=np.float32)

    if not np.isfinite(X_test).all():
        raise ValueError(
            f"{scenario}: Test contains NaN or infinity after frozen preprocessing."
        )

    model = keras.models.load_model(paths["model_path"])

    if int(model.input_shape[-1]) != int(cfg["k"]):
        raise RuntimeError(
            f"{scenario}: model input dimension {model.input_shape[-1]} "
            f"does not equal frozen K={cfg['k']}."
        )

    predict_start = time.perf_counter()
    y_pred, probabilities = predict_probabilities(
        model,
        X_test,
        n_classes=n_classes,
    )
    predict_seconds = time.perf_counter() - predict_start

    metrics = calculate_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        n_classes=n_classes,
    )

    # Per-sample predictions: this is the single source of truth for all metrics.
    prediction_frame = pd.DataFrame(
        {
            "Row_Index": np.arange(len(test_frame)),
            "True_Encoded": y_true,
            "Predicted_Encoded": y_pred,
            "True_Label": [class_names[i] for i in y_true],
            "Predicted_Label": [class_names[i] for i in y_pred],
        }
    )

    sample_id_col = choose_first_existing(test_frame.columns, SAMPLE_ID_CANDIDATES)
    if sample_id_col is not None:
        prediction_frame.insert(
            1,
            sample_id_col,
            test_frame[sample_id_col].astype(str).to_numpy(),
        )

    group_id_col = choose_first_existing(test_frame.columns, GROUP_ID_CANDIDATES)
    if group_id_col is not None and group_id_col not in prediction_frame.columns:
        prediction_frame.insert(
            2 if sample_id_col is not None else 1,
            group_id_col,
            test_frame[group_id_col].astype(str).to_numpy(),
        )

    for class_index, class_name in enumerate(class_names):
        prediction_frame[f"Probability_{class_name}"] = probabilities[:, class_index]

    prediction_path = scenario_output / "Final_Test_Predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    # Confusion matrix counts.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
    )
    cm_frame = pd.DataFrame(
        cm,
        index=[f"True_{x}" for x in class_names],
        columns=[f"Pred_{x}" for x in class_names],
    )
    cm_frame.to_excel(
        scenario_output / "Final_Test_Confusion_Matrix.xlsx"
    )

    # Class-wise metrics.
    classwise = classwise_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )
    classwise.to_excel(
        scenario_output / "Final_Test_Classwise_Metrics.xlsx",
        index=False,
    )

    # Bootstrap CIs.
    group_values = (
        test_frame[group_id_col].astype(str).to_numpy()
        if group_id_col is not None
        else None
    )
    bootstrap_frame, bootstrap_unit = bootstrap_distribution(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        n_classes=n_classes,
        group_values=group_values,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    if not bootstrap_frame.empty:
        bootstrap_frame.to_csv(
            scenario_output / "Final_Test_Bootstrap_Distribution.csv",
            index=False,
        )

    ci_frame = confidence_interval_summary(
        bootstrap_frame=bootstrap_frame,
        point_metrics=metrics,
    )
    ci_frame.to_excel(
        scenario_output / "Final_Test_Metrics_With_95CI.xlsx",
        index=False,
    )

    # Exact metrics in machine-readable form.
    metrics_payload = {
        "scenario": scenario,
        "patch_size": int(cfg["patch_size"]),
        "feature_method": str(cfg["feature_method"]),
        "k": int(cfg["k"]),
        "final_seed": int(cfg["seed"]),
        "source_variant": SOURCE_VARIANT,
        "test_rows": int(len(test_frame)),
        "class_names": class_names,
        "metrics": metrics,
        "bootstrap": {
            "repeats": int(bootstrap_repeats),
            "seed": int(bootstrap_seed),
            "unit": bootstrap_unit,
            "group_column": group_id_col,
        },
        "predict_seconds": float(predict_seconds),
    }
    json_dump(
        scenario_output / "Final_Test_Metrics.json",
        metrics_payload,
    )

    # Reproducibility / integrity metadata.
    checksums = {
        "model_sha256": sha256_file(paths["model_path"]),
        "selected_features_sha256": sha256_file(paths["selected_feature_path"]),
        "validation_run_metadata_sha256": sha256_file(paths["run_metadata_path"]),
        "test_input_sha256": sha256_file(paths["test_path"]),
        "final_predictions_sha256": sha256_file(prediction_path),
    }

    audit_payload = {
        "generated_at": utc_safe_timestamp(),
        "frozen_config_signature": frozen_config_signature(),
        "scenario": scenario,
        "frozen_config": cfg,
        "paths": {key: str(value) for key, value in paths.items()},
        "validation_artifact_checks": validation_check["validation_artifact_checks"],
        "selected_feature_count": len(selected_features),
        "test_rows": len(test_frame),
        "sample_id_column": sample_id_col,
        "group_id_column": group_id_col,
        "checksums": checksums,
        "software": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": getattr(keras, "__version__", "tensorflow.keras"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "scientific_policy": (
            "Frozen validation-selected configuration; fixed seed 42; "
            "no fitting, tuning, ranking, or configuration selection on Test."
        ),
    }
    json_dump(
        scenario_output / "Final_Test_Audit_Metadata.json",
        audit_payload,
    )

    del model, X_test, y_true, y_pred, probabilities

    return {
        "Scenario": scenario,
        "Patch_Size": int(cfg["patch_size"]),
        "Feature_Method": str(cfg["feature_method"]),
        "K": int(cfg["k"]),
        "Seed": int(cfg["seed"]),
        "Test_Rows": int(len(test_frame)),
        "Predict_Seconds": float(predict_seconds),
        **metrics,
    }


# =============================================================================
# Checkpoint / lock handling
# =============================================================================

CHECKPOINT_COLUMNS = [
    "Scenario",
    "Patch_Size",
    "Feature_Method",
    "K",
    "Seed",
    "Test_Rows",
    "Predict_Seconds",
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Precision_Weighted",
    "Recall_Weighted",
    "F1_Weighted",
    "ROC_AUC_OVR_Macro",
]


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    frame = pd.read_csv(path)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[CHECKPOINT_COLUMNS]


def save_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.stem + "_TEMP.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-way final locked DNN test for frozen MC-CRoMD configurations."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
    )
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--allow-completed-rerun",
        action="store_true",
        help=(
            "Administrative recovery only. Do not use this to search for better Test results."
        ),
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignored unknown arguments:", unknown)

    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats cannot be negative.")

    base_dir = Path(args.base_dir)
    output_dir = base_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    started_lock = output_dir / "FINAL_TEST_STARTED.lock"
    completed_lock = output_dir / "FINAL_TEST_COMPLETED.lock"
    checkpoint_path = output_dir / "Final_DNN_Locked_Test_Checkpoint.csv"

    if completed_lock.exists() and not args.allow_completed_rerun:
        raise RuntimeError(
            "FINAL_TEST_COMPLETED.lock already exists.\n"
            "The final locked test has already been completed. "
            "This program refuses to re-run it by default."
        )

    signature = frozen_config_signature()

    # Preflight BEFORE opening Test.
    preflight_rows = []
    for scenario in SCENARIO_ORDER:
        cfg = FROZEN_CONFIGS[scenario]
        paths = paths_for_config(base_dir, scenario)

        # Note: paths_for_config locates Test filename but does not read its contents.
        for key in (
            "selected_feature_path",
            "model_path",
            "run_metadata_path",
            "validation_selected_features_path",
            "test_path",
        ):
            if not paths[key].exists():
                raise FileNotFoundError(
                    f"Preflight failed for {scenario}: missing {key}: {paths[key]}"
                )

        selected = read_selected_features(
            paths["selected_feature_path"],
            expected_k=int(cfg["k"]),
        )
        validation_check = verify_against_validation_artifacts(
            paths=paths,
            scenario=scenario,
            selected_features=selected,
        )

        preflight_rows.append(
            {
                "Scenario": scenario,
                "Patch_Size": int(cfg["patch_size"]),
                "Feature_Method": str(cfg["feature_method"]),
                "K": int(cfg["k"]),
                "Seed": int(cfg["seed"]),
                "Model_Path": str(paths["model_path"]),
                "Test_Path": str(paths["test_path"]),
                "Validation_Artifact_Checks": json.dumps(
                    validation_check["validation_artifact_checks"],
                    sort_keys=True,
                ),
            }
        )

    pd.DataFrame(preflight_rows).to_excel(
        output_dir / "Final_Test_Preflight.xlsx",
        index=False,
    )

    preflight_metadata = {
        "generated_at": utc_safe_timestamp(),
        "frozen_config_signature": signature,
        "frozen_configs": FROZEN_CONFIGS,
        "final_seed_policy": (
            "Seed 42 was fixed before Test and is not selected according to Test performance."
        ),
        "selection_policy": (
            "Patch size, feature method, K, architecture, hyperparameters, and final seed "
            "are frozen before Test. Test is used only for final evaluation."
        ),
        "bootstrap_repeats": int(args.bootstrap_repeats),
        "bootstrap_seed": int(args.bootstrap_seed),
    }
    json_dump(
        output_dir / "Final_Test_Preflight_Metadata.json",
        preflight_metadata,
    )

    if not started_lock.exists():
        started_lock.write_text(
            json.dumps(
                {
                    "started_at": utc_safe_timestamp(),
                    "frozen_config_signature": signature,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 96)
    print("MC-CRoMD DNN FINAL LOCKED TEST")
    print("=" * 96)
    print("Frozen config signature :", signature)
    print("Final seed              :", FINAL_SEED)
    print("Bootstrap repeats       :", args.bootstrap_repeats)
    print("Output                   :", output_dir)
    print("IMPORTANT                : NO selection or tuning is permitted from this point.")
    print("=" * 96)

    checkpoint = load_checkpoint(checkpoint_path)
    completed_scenarios = set(
        checkpoint["Scenario"].dropna().astype(str)
    )

    for scenario in SCENARIO_ORDER:
        if scenario in completed_scenarios:
            print("Skip completed final scenario:", scenario)
            continue

        cfg = FROZEN_CONFIGS[scenario]
        print("\n" + "-" * 96)
        print(
            f"FINAL TEST: {scenario} | patch={cfg['patch_size']} | "
            f"{cfg['feature_method']} | K={cfg['k']} | seed={cfg['seed']}"
        )

        result = evaluate_one_scenario(
            base_dir=base_dir,
            output_dir=output_dir,
            scenario=scenario,
            bootstrap_repeats=int(args.bootstrap_repeats),
            bootstrap_seed=int(args.bootstrap_seed),
        )

        checkpoint = pd.concat(
            [
                checkpoint[checkpoint["Scenario"].astype(str) != scenario],
                pd.DataFrame([result], columns=CHECKPOINT_COLUMNS),
            ],
            ignore_index=True,
        )
        save_checkpoint(checkpoint, checkpoint_path)

        print(
            "Completed | "
            f"Accuracy={result['Accuracy']:.10f} | "
            f"Macro-F1={result['F1_Macro']:.10f} | "
            f"AUC={result['ROC_AUC_OVR_Macro']:.10f}"
        )

    # Final consolidated report.
    checkpoint = load_checkpoint(checkpoint_path)
    missing_final = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario not in set(checkpoint["Scenario"].astype(str))
    ]
    if missing_final:
        raise RuntimeError(
            "Final test did not complete all frozen scenarios: "
            + ", ".join(missing_final)
        )

    checkpoint = checkpoint.set_index("Scenario").loc[list(SCENARIO_ORDER)].reset_index()
    checkpoint.to_excel(
        output_dir / "Final_DNN_Locked_Test_Results.xlsx",
        index=False,
    )

    final_metadata = {
        "completed_at": utc_safe_timestamp(),
        "frozen_config_signature": signature,
        "frozen_configs": FROZEN_CONFIGS,
        "final_seed": FINAL_SEED,
        "source_variant": SOURCE_VARIANT,
        "results_file": str(output_dir / "Final_DNN_Locked_Test_Results.xlsx"),
        "checkpoint_file": str(checkpoint_path),
        "scientific_statement": (
            "All final DNN configuration decisions were frozen before Test. "
            "The Test partition was used only for final evaluation and did not "
            "influence preprocessing fitting, feature ranking, K selection, patch-size "
            "selection, DNN architecture, hyperparameters, or random-seed selection."
        ),
    }
    json_dump(
        output_dir / "Final_DNN_Locked_Test_Metadata.json",
        final_metadata,
    )

    completed_lock.write_text(
        json.dumps(
            {
                "completed_at": utc_safe_timestamp(),
                "frozen_config_signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 96)
    print("FINAL LOCKED TEST COMPLETED SUCCESSFULLY")
    print("=" * 96)
    print("Results :", output_dir / "Final_DNN_Locked_Test_Results.xlsx")
    print("Metadata:", output_dir / "Final_DNN_Locked_Test_Metadata.json")
    print("Lock    :", completed_lock)
    print("IMPORTANT: Do NOT change the frozen configurations based on these Test results.")
    print("=" * 96)


if __name__ == "__main__":
    main()
