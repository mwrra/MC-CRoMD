# -*- coding: utf-8 -*-
"""
Final_ML_Locked_Test.py

Final locked-test evaluation for the revised MC-CRoMD classical ML workflow.

SCIENTIFIC POLICY
-----------------
1) The three final ML configurations are HARD-CODED below and were frozen
   BEFORE loading Test, using only completed Tuning results from Validation.
2) The program verifies those frozen decisions against the Pre6 checkpoints
   for patch sizes 32, 64, 128, and 256 BEFORE opening Test.
3) It does NOT search, rank, tune, compare, or replace any configuration after
   Test is opened.
4) The selected feature list is loaded from the frozen Pre5 artifact.
5) The final classifier is fitted ONCE on the already-preprocessed Train split
   using the frozen hyperparameters. Validation is not used for fitting here.
6) No scaler, imputer, outlier rule, feature selector, threshold, or model
   hyperparameter is fitted/selected in this program.
7) Test is loaded only after all frozen-decision checks pass.
8) All reported metrics are derived from the same per-sample predictions.
9) Macro and weighted metrics, class-wise metrics, confusion matrix, and 95%
   bootstrap confidence intervals are saved.
10) Bootstrap is group-aware using SourceGroupID when available; otherwise it
    falls back to sample-level bootstrap.
11) Checkpoint and lock files protect the one-way final evaluation workflow.
12) Re-running after FINAL_ML_TEST_COMPLETED.lock exists is refused by default.

FROZEN ML CONFIGURATIONS
------------------------
Binary:
    patch 128 | Logistic | K=125 | Voting(hard)
    Validation Accuracy = 0.9849624060150376
    Validation Macro-F1 = 0.9849615558570782

Three_Class:
    patch 256 | Variance | K=200 | SVM_RBF
    C=10.0, gamma='scale'
    Validation Accuracy = 0.9297153024911032
    Validation Macro-F1 = 0.9311504488293526

Five_Class:
    patch 256 | Logistic | K=200 | LogisticRegression
    C=10.0, penalty='l2'
    Validation Accuracy = 0.9396462018730489
    Validation Macro-F1 = 0.93978824070617

Selection criterion:
    Validation Accuracy first, then Validation Macro-F1.

IMPORTANT
---------
Do not change any frozen configuration after observing Test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Frozen scientific decisions -- DO NOT MODIFY AFTER TEST IS OPENED
# =============================================================================

RANDOM_STATE = 42
TARGET_COLUMN = "label"
PRIMARY_METRIC = "Validation_Accuracy"
SECONDARY_METRIC = "Validation_F1_Macro"

FROZEN_CONFIGS: dict[str, dict[str, Any]] = {
    "Binary": {
        "patch_size": 128,
        "feature_method": "Logistic",
        "k": 125,
        "model": "Voting",
        "source_variant": "StandardScaled",
        "parameters": {"voting": "hard"},
        "validation_accuracy": 0.9849624060150376,
        "validation_f1_macro": 0.9849615558570782,
    },
    "Three_Class": {
        "patch_size": 256,
        "feature_method": "Variance",
        "k": 200,
        "model": "SVM_RBF",
        "source_variant": "StandardScaled",
        "parameters": {"C": 10.0, "gamma": "scale"},
        "validation_accuracy": 0.9297153024911032,
        "validation_f1_macro": 0.9311504488293526,
    },
    "Five_Class": {
        "patch_size": 256,
        "feature_method": "Logistic",
        "k": 200,
        "model": "LogisticRegression",
        "source_variant": "StandardScaled",
        "parameters": {"C": 10.0, "penalty": "l2"},
        "validation_accuracy": 0.9396462018730489,
        "validation_f1_macro": 0.93978824070617,
    },
}

SCENARIO_ORDER = ("Binary", "Three_Class", "Five_Class")
PATCH_SIZES = (32, 64, 128, 256)

DEFAULT_BASE_DIR = Path(r"G:\My Research About Lung Canser\INASS")
DEFAULT_OUTPUT_DIR_NAME = "8-MLFinalLockedTest"
DEFAULT_BOOTSTRAP_REPEATS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260819

SAMPLE_ID_CANDIDATES = ("ImageSHA256", "FileName", "ImgName")
GROUP_ID_CANDIDATES = ("SourceGroupID", "LC25000_GroupID", "group_id")


# =============================================================================
# Utilities
# =============================================================================

def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def frozen_config_signature() -> str:
    payload = json.dumps(
        {
            "random_state": RANDOM_STATE,
            "criterion": [PRIMARY_METRIC, SECONDARY_METRIC],
            "configs": FROZEN_CONFIGS,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        "Could not find any of:\n" + "\n".join(str(x) for x in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def choose_first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def canonical_json_text(value: Any) -> str:
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# =============================================================================
# Frozen-decision verification against Pre6 (BEFORE Test)
# =============================================================================

def load_all_tuning_results(base_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for patch in PATCH_SIZES:
        path = base_dir / "5-ModelValidation" / str(patch) / "Validation_Checkpoint.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Pre6 checkpoint required for final-decision verification: {path}"
            )

        frame = pd.read_csv(path)
        required = {
            "Stage", "Scenario", "Feature_Method", "K", "Model",
            "Parameters_JSON", PRIMARY_METRIC, SECONDARY_METRIC, "Status",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{path}: missing columns {sorted(missing)}")

        frame = frame[
            frame["Stage"].astype(str).str.strip().eq("Tuning")
            & frame["Status"].astype(str).str.strip().eq("Completed")
        ].copy()

        frame["Patch_Size"] = int(patch)
        frame["K"] = pd.to_numeric(frame["K"], errors="raise").astype(int)
        frame[PRIMARY_METRIC] = pd.to_numeric(frame[PRIMARY_METRIC], errors="raise")
        frame[SECONDARY_METRIC] = pd.to_numeric(frame[SECONDARY_METRIC], errors="raise")
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("No completed Tuning results were found across Pre6 checkpoints.")
    return combined


def verify_frozen_configs_are_global_validation_winners(
    tuning: pd.DataFrame,
) -> pd.DataFrame:
    audit_rows: list[dict[str, Any]] = []

    for scenario in SCENARIO_ORDER:
        cfg = FROZEN_CONFIGS[scenario]
        subset = tuning[tuning["Scenario"].astype(str).eq(scenario)].copy()
        if subset.empty:
            raise RuntimeError(f"No completed Tuning rows found for {scenario}.")

        subset = subset.sort_values(
            [PRIMARY_METRIC, SECONDARY_METRIC, "Patch_Size", "Model", "Feature_Method", "K"],
            ascending=[False, False, True, True, True, True],
            kind="mergesort",
        )
        winner = subset.iloc[0]

        expected_params = canonical_json_text(cfg["parameters"])
        actual_params = canonical_json_text(winner["Parameters_JSON"])

        checks = {
            "patch_size": int(winner["Patch_Size"]) == int(cfg["patch_size"]),
            "feature_method": str(winner["Feature_Method"]).strip() == str(cfg["feature_method"]),
            "k": int(winner["K"]) == int(cfg["k"]),
            "model": str(winner["Model"]).strip() == str(cfg["model"]),
            "parameters": actual_params == expected_params,
            "validation_accuracy": np.isclose(
                float(winner[PRIMARY_METRIC]), float(cfg["validation_accuracy"]), rtol=0, atol=1e-15
            ),
            "validation_f1_macro": np.isclose(
                float(winner[SECONDARY_METRIC]), float(cfg["validation_f1_macro"]), rtol=0, atol=1e-15
            ),
        }

        if not all(checks.values()):
            raise RuntimeError(
                f"{scenario}: hard-coded frozen configuration is NOT the current global "
                f"Validation/Tuning winner. Checks={checks}\n"
                f"Current winner: patch={winner['Patch_Size']}, "
                f"method={winner['Feature_Method']}, K={winner['K']}, "
                f"model={winner['Model']}, params={winner['Parameters_JSON']}, "
                f"acc={winner[PRIMARY_METRIC]}, f1={winner[SECONDARY_METRIC]}"
            )

        audit_rows.append(
            {
                "Scenario": scenario,
                "Patch_Size": int(cfg["patch_size"]),
                "Feature_Method": str(cfg["feature_method"]),
                "K": int(cfg["k"]),
                "Model": str(cfg["model"]),
                "Parameters_JSON": json.dumps(cfg["parameters"], ensure_ascii=False, sort_keys=True),
                "Validation_Accuracy": float(cfg["validation_accuracy"]),
                "Validation_F1_Macro": float(cfg["validation_f1_macro"]),
                "Verified_Global_Winner": True,
            }
        )

    return pd.DataFrame(audit_rows)


# =============================================================================
# Frozen paths / selected features
# =============================================================================

def paths_for_config(base_dir: Path, scenario: str) -> dict[str, Path]:
    cfg = FROZEN_CONFIGS[scenario]
    patch = int(cfg["patch_size"])
    method = str(cfg["feature_method"])
    k = int(cfg["k"])
    variant = str(cfg["source_variant"])

    preprocessing_dir = base_dir / "3-DataPreprocessing" / str(patch)
    feature_selection_dir = base_dir / "4-FeatureSelection" / str(patch)

    selected_feature_path = (
        feature_selection_dir
        / scenario
        / method
        / f"Selected_Features_K{k:03d}.xlsx"
    )

    variant_dir = preprocessing_dir / scenario / variant
    train_path = find_existing_file(variant_dir, "Train")
    test_path = find_existing_file(variant_dir, "Test")

    return {
        "preprocessing_dir": preprocessing_dir,
        "feature_selection_dir": feature_selection_dir,
        "selected_feature_path": selected_feature_path,
        "train_path": train_path,
        "test_path": test_path,
    }


def read_selected_features(path: Path, expected_k: int) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path)
    if "Feature" not in frame.columns:
        raise KeyError(f"'Feature' column not found in {path}")
    values = frame["Feature"].dropna().astype(str).str.strip().tolist()
    values = [x for x in values if x]
    if len(values) != expected_k:
        raise RuntimeError(
            f"{path}: expected K={expected_k}, but file contains {len(values)} features."
        )
    if len(values) != len(set(values)):
        raise RuntimeError(f"Duplicate selected features found in {path}")
    return values


# =============================================================================
# Model recreation -- identical scientific hyperparameters to Pre6
# =============================================================================

def create_model(model_name: str, params: dict[str, Any]):
    if model_name == "SVM_RBF":
        return SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            cache_size=2000,
        )

    if model_name == "SVM_Poly":
        return SVC(
            kernel="poly",
            C=params["C"],
            degree=params["degree"],
            gamma=params["gamma"],
            coef0=params["coef0"],
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            cache_size=2000,
        )

    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            p=params["p"],
            n_jobs=-1,
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            max_features=params["max_features"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            criterion=params["criterion"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "LogisticRegression":
        penalty = params["penalty"]
        solver = "saga" if penalty == "l1" else "lbfgs"
        return LogisticRegression(
            C=params["C"],
            penalty=penalty,
            solver=solver,
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "SGD":
        return SGDClassifier(
            loss=params["loss"],
            alpha=params["alpha"],
            penalty=params["penalty"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=3000,
            tol=1e-4,
            early_stopping=False,
        )

    if model_name == "GaussianNB":
        return GaussianNB(var_smoothing=params["var_smoothing"])

    if model_name == "Voting":
        estimators = [
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            ("gnb", GaussianNB(var_smoothing=1e-9)),
        ]
        return VotingClassifier(
            estimators=estimators,
            voting=params["voting"],
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Metrics
# =============================================================================

def get_score_matrix(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X), dtype=float)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X), dtype=float)
        except Exception:
            pass
    return None


def calculate_auc(
    y_true: np.ndarray,
    score_matrix: np.ndarray | None,
    classes: np.ndarray,
) -> float:
    if score_matrix is None:
        return float("nan")
    try:
        if len(classes) == 2:
            if score_matrix.ndim == 2:
                positive_scores = score_matrix[:, 1]
            else:
                positive_scores = score_matrix
            return float(roc_auc_score(y_true, positive_scores))

        if score_matrix.ndim == 1:
            return float("nan")
        y_binary = label_binarize(y_true, classes=classes)
        return float(
            roc_auc_score(
                y_binary,
                score_matrix,
                average="macro",
                multi_class="ovr",
            )
        )
    except Exception:
        return float("nan")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_matrix: np.ndarray | None,
    classes: np.ndarray,
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
        "ROC_AUC_OVR_Macro": calculate_auc(y_true, score_matrix, classes),
    }


def classwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_indices: np.ndarray,
    display_names: list[str],
) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_indices,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "Class_Index": class_indices,
            "Class_Name": display_names,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support.astype(int),
        }
    )


# =============================================================================
# Bootstrap CIs
# =============================================================================

def bootstrap_distribution(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_matrix: np.ndarray | None,
    classes: np.ndarray,
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
        unit = "SourceGroupID"
    else:
        unique_groups = None
        group_to_indices = None
        unit = "sample"

    for iteration in range(1, repeats + 1):
        if unique_groups is not None:
            sampled_groups = rng.choice(
                unique_groups,
                size=len(unique_groups),
                replace=True,
            )
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))

        sampled_scores = None if score_matrix is None else score_matrix[idx]
        metrics = calculate_metrics(
            y_true[idx],
            y_pred[idx],
            sampled_scores,
            classes,
        )
        rows.append({"Iteration": iteration, **metrics})

    return pd.DataFrame(rows), unit


def confidence_interval_summary(
    bootstrap_frame: pd.DataFrame,
    point_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for metric, point in point_metrics.items():
        if bootstrap_frame.empty or metric not in bootstrap_frame.columns:
            low = high = float("nan")
        else:
            values = pd.to_numeric(bootstrap_frame[metric], errors="coerce").dropna()
            if values.empty:
                low = high = float("nan")
            else:
                low = float(np.quantile(values, 0.025))
                high = float(np.quantile(values, 0.975))
        rows.append(
            {
                "Metric": metric,
                "Point_Estimate": float(point),
                "CI_95_Lower": low,
                "CI_95_Upper": high,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# One frozen final scenario
# =============================================================================

def build_display_class_names(
    train_frame: pd.DataFrame,
    encoder: LabelEncoder,
) -> list[str]:
    if "ClassName" not in train_frame.columns:
        return [str(x) for x in encoder.classes_]

    temp = train_frame[[TARGET_COLUMN, "ClassName"]].dropna().copy()
    temp[TARGET_COLUMN] = temp[TARGET_COLUMN].astype(str)
    temp["ClassName"] = temp["ClassName"].astype(str)

    mapping: dict[str, str] = {}
    for label, group in temp.groupby(TARGET_COLUMN):
        names = sorted(group["ClassName"].unique())
        if len(names) == 1:
            mapping[str(label)] = names[0]

    return [mapping.get(str(x), str(x)) for x in encoder.classes_]


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

    # Train is loaded first. Test remains unopened until the frozen model has been built.
    train_frame = read_table(paths["train_path"])
    train_frame.columns = [str(c).strip() for c in train_frame.columns]

    if TARGET_COLUMN not in train_frame.columns:
        raise KeyError(f"{scenario}: Train is missing target column {TARGET_COLUMN!r}.")
    if "Split" in train_frame.columns:
        values = set(train_frame["Split"].astype(str).str.strip().unique())
        if values != {"Train"}:
            raise RuntimeError(f"{scenario}: expected Train only; found {sorted(values)}")

    missing_train_features = sorted(set(selected_features).difference(train_frame.columns))
    if missing_train_features:
        raise KeyError(
            f"{scenario}: Train is missing {len(missing_train_features)} frozen features. "
            f"Examples: {missing_train_features[:10]}"
        )

    X_train = train_frame[selected_features].to_numpy(dtype=np.float32)
    if not np.isfinite(X_train).all():
        raise ValueError(f"{scenario}: Train contains NaN/inf after frozen preprocessing.")

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_frame[TARGET_COLUMN].astype(str).to_numpy())
    classes = np.arange(len(encoder.classes_))
    display_class_names = build_display_class_names(train_frame, encoder)

    model = create_model(str(cfg["model"]), dict(cfg["parameters"]))

    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    # Persist the exactly-fitted frozen final model BEFORE Test evaluation.
    model_path = scenario_output / "Frozen_Final_Train_Model.joblib"
    joblib.dump(
        {
            "model": model,
            "label_encoder": encoder,
            "selected_features": selected_features,
            "frozen_config": cfg,
            "fit_seconds": fit_seconds,
        },
        model_path,
        compress=3,
    )

    # -------------------------------------------------------------------------
    # TEST IS OPENED ONLY HERE, after all selection and fitting decisions ended.
    # -------------------------------------------------------------------------
    test_frame = read_table(paths["test_path"])
    test_frame.columns = [str(c).strip() for c in test_frame.columns]

    if TARGET_COLUMN not in test_frame.columns:
        raise KeyError(f"{scenario}: Test is missing target column {TARGET_COLUMN!r}.")
    if "Split" in test_frame.columns:
        values = set(test_frame["Split"].astype(str).str.strip().unique())
        if values != {"Test"}:
            raise RuntimeError(f"{scenario}: expected Test only; found {sorted(values)}")

    missing_test_features = sorted(set(selected_features).difference(test_frame.columns))
    if missing_test_features:
        raise KeyError(
            f"{scenario}: Test is missing {len(missing_test_features)} frozen features. "
            f"Examples: {missing_test_features[:10]}"
        )

    raw_test_labels = test_frame[TARGET_COLUMN].astype(str).to_numpy()
    unknown = sorted(set(raw_test_labels).difference(set(encoder.classes_.astype(str))))
    if unknown:
        raise ValueError(f"{scenario}: Test contains unseen labels: {unknown}")

    y_test = encoder.transform(raw_test_labels)
    X_test = test_frame[selected_features].to_numpy(dtype=np.float32)
    if not np.isfinite(X_test).all():
        raise ValueError(f"{scenario}: Test contains NaN/inf after frozen preprocessing.")

    predict_start = time.perf_counter()
    y_pred = model.predict(X_test)
    score_matrix = get_score_matrix(model, X_test)
    predict_seconds = time.perf_counter() - predict_start

    metrics = calculate_metrics(y_test, y_pred, score_matrix, classes)

    # Per-sample source of truth.
    prediction_frame = pd.DataFrame(
        {
            "Row_Index": np.arange(len(test_frame)),
            "True_Encoded": y_test,
            "Predicted_Encoded": y_pred,
            "True_Label": [display_class_names[i] for i in y_test],
            "Predicted_Label": [display_class_names[i] for i in y_pred],
        }
    )

    sample_col = choose_first_existing(test_frame.columns, SAMPLE_ID_CANDIDATES)
    if sample_col is not None:
        prediction_frame.insert(1, sample_col, test_frame[sample_col].astype(str).to_numpy())

    group_col = choose_first_existing(test_frame.columns, GROUP_ID_CANDIDATES)
    if group_col is not None and group_col not in prediction_frame.columns:
        prediction_frame.insert(
            2 if sample_col is not None else 1,
            group_col,
            test_frame[group_col].astype(str).to_numpy(),
        )

    # Save scores only when the selected estimator exposes them.
    if score_matrix is not None:
        if score_matrix.ndim == 1:
            prediction_frame["Decision_Score"] = score_matrix
        elif score_matrix.ndim == 2 and score_matrix.shape[1] == len(classes):
            for i, name in enumerate(display_class_names):
                prediction_frame[f"Score_{name}"] = score_matrix[:, i]

    predictions_path = scenario_output / "Final_ML_Test_Predictions.csv"
    prediction_frame.to_csv(predictions_path, index=False)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    pd.DataFrame(
        cm,
        index=[f"True_{x}" for x in display_class_names],
        columns=[f"Pred_{x}" for x in display_class_names],
    ).to_excel(scenario_output / "Final_ML_Test_Confusion_Matrix.xlsx")

    classwise_metrics(
        y_test,
        y_pred,
        classes,
        display_class_names,
    ).to_excel(
        scenario_output / "Final_ML_Test_Classwise_Metrics.xlsx",
        index=False,
    )

    group_values = (
        test_frame[group_col].astype(str).to_numpy()
        if group_col is not None
        else None
    )
    bootstrap_frame, bootstrap_unit = bootstrap_distribution(
        y_true=y_test,
        y_pred=y_pred,
        score_matrix=score_matrix,
        classes=classes,
        group_values=group_values,
        repeats=bootstrap_repeats,
        seed=bootstrap_seed,
    )
    if not bootstrap_frame.empty:
        bootstrap_frame.to_csv(
            scenario_output / "Final_ML_Test_Bootstrap_Distribution.csv",
            index=False,
        )

    confidence_interval_summary(
        bootstrap_frame,
        metrics,
    ).to_excel(
        scenario_output / "Final_ML_Test_Metrics_With_95CI.xlsx",
        index=False,
    )

    metrics_payload = {
        "scenario": scenario,
        "frozen_config": cfg,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "class_names": display_class_names,
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "metrics": metrics,
        "auc_note": (
            "ROC-AUC is unavailable (NaN) when the frozen estimator does not expose "
            "predict_proba or decision_function, e.g. hard Voting."
        ),
        "bootstrap": {
            "repeats": int(bootstrap_repeats),
            "seed": int(bootstrap_seed),
            "unit": bootstrap_unit,
            "group_column": group_col,
        },
    }
    json_dump(scenario_output / "Final_ML_Test_Metrics.json", metrics_payload)

    audit_payload = {
        "generated_at": timestamp(),
        "frozen_config_signature": frozen_config_signature(),
        "scenario": scenario,
        "frozen_config": cfg,
        "paths": {key: str(value) for key, value in paths.items()},
        "selected_feature_count": len(selected_features),
        "sample_id_column": sample_col,
        "group_id_column": group_col,
        "checksums": {
            "selected_features_sha256": sha256_file(paths["selected_feature_path"]),
            "train_input_sha256": sha256_file(paths["train_path"]),
            "test_input_sha256": sha256_file(paths["test_path"]),
            "fitted_model_sha256": sha256_file(model_path),
            "final_predictions_sha256": sha256_file(predictions_path),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "scientific_policy": (
            "Configuration frozen from Validation/Tuning only; model fitted once on "
            "frozen preprocessed Train; Test opened only for final evaluation; no "
            "preprocessing fitting, feature selection, hyperparameter tuning, model "
            "selection, or threshold tuning on Test."
        ),
    }
    json_dump(scenario_output / "Final_ML_Test_Audit_Metadata.json", audit_payload)

    return {
        "Scenario": scenario,
        "Patch_Size": int(cfg["patch_size"]),
        "Feature_Method": str(cfg["feature_method"]),
        "K": int(cfg["k"]),
        "Model": str(cfg["model"]),
        "Parameters_JSON": json.dumps(cfg["parameters"], ensure_ascii=False, sort_keys=True),
        "Train_Rows": int(len(train_frame)),
        "Test_Rows": int(len(test_frame)),
        "Fit_Seconds": float(fit_seconds),
        "Predict_Seconds": float(predict_seconds),
        **metrics,
    }


# =============================================================================
# Checkpoint / lock handling
# =============================================================================

CHECKPOINT_COLUMNS = [
    "Scenario", "Patch_Size", "Feature_Method", "K", "Model",
    "Parameters_JSON", "Train_Rows", "Test_Rows", "Fit_Seconds",
    "Predict_Seconds", "Accuracy", "Precision_Macro", "Recall_Macro",
    "F1_Macro", "Precision_Weighted", "Recall_Weighted", "F1_Weighted",
    "ROC_AUC_OVR_Macro",
]


def load_final_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)
    frame = pd.read_csv(path)
    for column in CHECKPOINT_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[CHECKPOINT_COLUMNS]


def save_final_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.stem + "_TEMP.csv")
    frame.to_csv(temp, index=False)
    temp.replace(path)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-way final locked ML test for frozen MC-CRoMD configurations."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
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
        help="Administrative recovery only; never use to search for better Test results.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignored unknown arguments:", unknown)

    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats cannot be negative.")

    base_dir = Path(args.base_dir)
    output_dir = base_dir / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    started_lock = output_dir / "FINAL_ML_TEST_STARTED.lock"
    completed_lock = output_dir / "FINAL_ML_TEST_COMPLETED.lock"
    checkpoint_path = output_dir / "Final_ML_Locked_Test_Checkpoint.csv"

    if completed_lock.exists() and not args.allow_completed_rerun:
        raise RuntimeError(
            "FINAL_ML_TEST_COMPLETED.lock already exists.\n"
            "The final ML locked test has already completed. Re-run is refused by default."
        )

    signature = frozen_config_signature()

    # -------------------------------------------------------------------------
    # PRE-FLIGHT: all decision verification happens BEFORE Test is read.
    # -------------------------------------------------------------------------
    tuning = load_all_tuning_results(base_dir)
    winners = verify_frozen_configs_are_global_validation_winners(tuning)

    preflight_rows = []
    for scenario in SCENARIO_ORDER:
        cfg = FROZEN_CONFIGS[scenario]
        paths = paths_for_config(base_dir, scenario)  # locates Test path; does not read Test
        for key in ("selected_feature_path", "train_path", "test_path"):
            if not paths[key].exists():
                raise FileNotFoundError(f"{scenario}: missing {key}: {paths[key]}")
        selected = read_selected_features(
            paths["selected_feature_path"],
            int(cfg["k"]),
        )
        preflight_rows.append(
            {
                "Scenario": scenario,
                "Patch_Size": int(cfg["patch_size"]),
                "Feature_Method": str(cfg["feature_method"]),
                "K": int(cfg["k"]),
                "Model": str(cfg["model"]),
                "Parameters_JSON": json.dumps(cfg["parameters"], ensure_ascii=False, sort_keys=True),
                "Source_Variant": str(cfg["source_variant"]),
                "Selected_Feature_Count": len(selected),
                "Validation_Accuracy": float(cfg["validation_accuracy"]),
                "Validation_F1_Macro": float(cfg["validation_f1_macro"]),
                "Verified_Global_Validation_Winner": True,
                "Train_Path": str(paths["train_path"]),
                "Test_Path": str(paths["test_path"]),
            }
        )

    pd.DataFrame(preflight_rows).to_excel(
        output_dir / "Final_ML_Test_Preflight.xlsx",
        index=False,
    )
    winners.to_excel(
        output_dir / "Frozen_ML_Validation_Winners.xlsx",
        index=False,
    )

    json_dump(
        output_dir / "Final_ML_Test_Preflight_Metadata.json",
        {
            "generated_at": timestamp(),
            "frozen_config_signature": signature,
            "frozen_configs": FROZEN_CONFIGS,
            "selection_criterion": (
                "Highest completed Tuning Validation Accuracy across patch sizes 32/64/128/256; "
                "Validation Macro-F1 is the secondary criterion."
            ),
            "random_state": RANDOM_STATE,
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "bootstrap_seed": int(args.bootstrap_seed),
            "test_policy": "Test is not read during configuration verification or selection.",
        },
    )

    if not started_lock.exists():
        started_lock.write_text(
            json.dumps(
                {
                    "started_at": timestamp(),
                    "frozen_config_signature": signature,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("=" * 100)
    print("MC-CRoMD ML FINAL LOCKED TEST")
    print("=" * 100)
    print("Frozen config signature :", signature)
    print("Random state            :", RANDOM_STATE)
    print("Bootstrap repeats       :", args.bootstrap_repeats)
    print("Output                  :", output_dir)
    print("IMPORTANT               : NO selection or tuning is permitted from this point.")
    print("=" * 100)

    checkpoint = load_final_checkpoint(checkpoint_path)
    completed_scenarios = set(checkpoint["Scenario"].dropna().astype(str))

    for scenario in SCENARIO_ORDER:
        if scenario in completed_scenarios:
            print("Skip completed final scenario:", scenario)
            continue

        cfg = FROZEN_CONFIGS[scenario]
        print("\n" + "-" * 100)
        print(
            f"FINAL ML TEST: {scenario} | patch={cfg['patch_size']} | "
            f"{cfg['feature_method']} | K={cfg['k']} | {cfg['model']} | "
            f"{cfg['parameters']}"
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
        save_final_checkpoint(checkpoint, checkpoint_path)

        auc = result["ROC_AUC_OVR_Macro"]
        auc_text = "NA" if pd.isna(auc) else f"{auc:.10f}"
        print(
            "Completed | "
            f"Accuracy={result['Accuracy']:.10f} | "
            f"Macro-F1={result['F1_Macro']:.10f} | "
            f"AUC={auc_text}"
        )

    checkpoint = load_final_checkpoint(checkpoint_path)
    missing = [
        scenario for scenario in SCENARIO_ORDER
        if scenario not in set(checkpoint["Scenario"].astype(str))
    ]
    if missing:
        raise RuntimeError("Final ML test incomplete for: " + ", ".join(missing))

    checkpoint = checkpoint.set_index("Scenario").loc[list(SCENARIO_ORDER)].reset_index()
    results_path = output_dir / "Final_ML_Locked_Test_Results.xlsx"
    checkpoint.to_excel(results_path, index=False)

    json_dump(
        output_dir / "Final_ML_Locked_Test_Metadata.json",
        {
            "completed_at": timestamp(),
            "frozen_config_signature": signature,
            "frozen_configs": FROZEN_CONFIGS,
            "selection_criterion": [PRIMARY_METRIC, SECONDARY_METRIC],
            "random_state": RANDOM_STATE,
            "results_file": str(results_path),
            "checkpoint_file": str(checkpoint_path),
            "scientific_statement": (
                "All final ML configuration decisions were frozen from completed Pre6 "
                "Tuning/Validation results before Test. Each frozen classifier was fitted "
                "once on the corresponding already-preprocessed Train split using the "
                "frozen selected features and hyperparameters. Test was used only for final "
                "evaluation and did not influence preprocessing, feature selection, patch-size "
                "selection, model choice, hyperparameters, or thresholds."
            ),
        },
    )

    completed_lock.write_text(
        json.dumps(
            {
                "completed_at": timestamp(),
                "frozen_config_signature": signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("FINAL ML LOCKED TEST COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print("Results :", results_path)
    print("Metadata:", output_dir / "Final_ML_Locked_Test_Metadata.json")
    print("Lock    :", completed_lock)
    print("IMPORTANT: Do NOT change frozen ML configurations based on Test results.")
    print("=" * 100)


if __name__ == "__main__":
    main()
