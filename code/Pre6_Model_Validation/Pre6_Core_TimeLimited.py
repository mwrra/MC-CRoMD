# -*- coding: utf-8 -*-
"""
Pre6_Model_Training_Validation_Spyder.py

تدريب النماذج واختيار أفضل:
- سيناريو
- طريقة اختيار خصائص
- عدد خصائص K
- نموذج
- إعدادات النموذج

باستخدام Train للتدريب وValidation للاختيار فقط.
لا يقرأ Test ولا يستخدمه في هذه المرحلة.

المدخلات:
1) مخرجات Pre4_Data_Preprocessing
2) مخرجات Pre5_Feature_Selection

النماذج الكلاسيكية:
- SVM_RBF
- SVM_Poly
- KNN
- RandomForest
- DecisionTree
- LogisticRegression
- SGD
- GaussianNB
- Voting

العمل على مرحلتين:
A) Screening:
   تقييم إعداد أساسي واحد لكل نموذج ولكل Method/K.
B) Tuning:
   أخذ أفضل TOP_N إعدادات Feature Selection لكل نموذج،
   ثم تجربة شبكة صغيرة من المعاملات.

ميزات مهمة:
- Checkpoint بعد كل تجربة.
- Resume تلقائي عند إعادة التشغيل.
- حفظ أفضل نموذج لكل سيناريو.
- حساب Accuracy وPrecision وRecall وF1 وROC-AUC.
- حفظ Confusion Matrix لأفضل النتائج.
- عدم استخدام Test إطلاقًا.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# =========================================================
# الإعدادات
# =========================================================

DEFAULT_PREPROCESSING_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\3-DataPreprocessing\256"
)

DEFAULT_FEATURE_SELECTION_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\4-FeatureSelection\256"
)

DEFAULT_OUTPUT_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\5-ModelValidation\256"
)

SCENARIOS = ("Binary", "Three_Class", "Five_Class")
VALID_SPLITS = ("Train", "Validation")

RANDOM_STATE = 42
TARGET_COLUMN = "label"

# عدد أفضل Method/K التي تدخل مرحلة الضبط لكل نموذج ولكل سيناريو.
TOP_N_FEATURE_CONFIGS_PER_MODEL = 3

# False = جميع قيم K الموجودة من Pre5.
# True = وضع أسرع للاختبار الأولي.
FAST_MODE = False
FAST_K_VALUES = {25, 50, 100, 150, 225}

# النماذج المطلوب تشغيلها.
ENABLED_MODELS = (
    "SVM_RBF",
    "SVM_Poly",
    "KNN",
    "RandomForest",
    "DecisionTree",
    "LogisticRegression",
    "SGD",
    "GaussianNB",
    "Voting",
)

# مقياس الاختيار الأساسي.
PRIMARY_METRIC = "Validation_Accuracy"
SECONDARY_METRIC = "Validation_F1_Macro"

METADATA_COLUMNS = {
    "ImgName",
    "FileName",
    "ImagePath",
    "ClassName",
    "label",
    "SourceGroupID",
    "PatchSize",
    "ImageWidth",
    "ImageHeight",
    "ImageSHA256",
    "Split",
    "split",
    "stem",
    "filename",
    "tissue",
    "group_id",
    "local_cluster_label",
    "LC25000_ClassName",
    "LC25000_FileName",
    "LC25000_Tissue",
    "Previous_SourceGroupID",
    "_match_stem_LC25000",
    "LC25000_GroupID",
    "LC25000_LocalCluster",
}

MODEL_SOURCE_VARIANT = {
    "SVM_RBF": "StandardScaled",
    "SVM_Poly": "StandardScaled",
    "KNN": "StandardScaled",
    "RandomForest": "Clean_Unscaled",
    "DecisionTree": "Clean_Unscaled",
    "LogisticRegression": "StandardScaled",
    "SGD": "StandardScaled",
    "GaussianNB": "Clean_Unscaled",
    "Voting": "StandardScaled",
}


# =========================================================
# شبكات المعاملات
# =========================================================

SCREENING_PARAMETERS: dict[str, dict[str, Any]] = {
    "SVM_RBF": {
        "C": 10.0,
        "gamma": "scale",
    },
    "SVM_Poly": {
        "C": 1.0,
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2,
    },
    "RandomForest": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
    },
    "DecisionTree": {
        "max_depth": None,
        "min_samples_leaf": 1,
        "criterion": "gini",
    },
    "LogisticRegression": {
        "C": 1.0,
        "penalty": "l2",
    },
    "SGD": {
        "loss": "log_loss",
        "alpha": 1e-4,
        "penalty": "l2",
    },
    "GaussianNB": {
        "var_smoothing": 1e-9,
    },
    "Voting": {
        "voting": "hard",
    },
}

TUNING_GRIDS: dict[str, list[dict[str, Any]]] = {
    "SVM_RBF": [
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 100.0, "gamma": "scale"},
    ],
    "SVM_Poly": [
        {"C": 1.0, "degree": 2, "gamma": "scale", "coef0": 0.0},
        {"C": 1.0, "degree": 3, "gamma": "scale", "coef0": 0.0},
        {"C": 10.0, "degree": 3, "gamma": "scale", "coef0": 1.0},
    ],
    "KNN": [
        {"n_neighbors": 3, "weights": "distance", "p": 2},
        {"n_neighbors": 5, "weights": "distance", "p": 2},
        {"n_neighbors": 7, "weights": "distance", "p": 2},
        {"n_neighbors": 9, "weights": "distance", "p": 2},
    ],
    "RandomForest": [
        {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 500,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 300,
            "max_depth": 30,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
        },
        {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": 0.5,
            "min_samples_leaf": 2,
        },
    ],
    "DecisionTree": [
        {"max_depth": None, "min_samples_leaf": 1, "criterion": "gini"},
        {"max_depth": 20, "min_samples_leaf": 1, "criterion": "gini"},
        {"max_depth": 30, "min_samples_leaf": 2, "criterion": "gini"},
        {"max_depth": 30, "min_samples_leaf": 1, "criterion": "entropy"},
    ],
    "LogisticRegression": [
        {"C": 0.1, "penalty": "l2"},
        {"C": 1.0, "penalty": "l2"},
        {"C": 10.0, "penalty": "l2"},
        {"C": 1.0, "penalty": "l1"},
    ],
    "SGD": [
        {"loss": "log_loss", "alpha": 1e-3, "penalty": "l2"},
        {"loss": "log_loss", "alpha": 1e-4, "penalty": "l2"},
        {"loss": "log_loss", "alpha": 1e-5, "penalty": "l2"},
        {"loss": "modified_huber", "alpha": 1e-4, "penalty": "l2"},
    ],
    "GaussianNB": [
        {"var_smoothing": 1e-7},
        {"var_smoothing": 1e-8},
        {"var_smoothing": 1e-9},
        {"var_smoothing": 1e-10},
    ],
    "Voting": [
        {"voting": "hard"},
    ],
}


# =========================================================
# وظائف الملفات والبيانات
# =========================================================

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
        "Could not find any of:\n" + "\n".join(str(item) for item in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def load_variant_split(
    preprocessing_dir: Path,
    scenario: str,
    variant: str,
    split_name: str,
) -> pd.DataFrame:
    folder = preprocessing_dir / scenario / variant
    path = find_existing_file(folder, split_name)
    frame = read_table(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def discover_feature_methods(
    feature_selection_dir: Path,
    scenario: str,
) -> dict[str, dict[int, Path]]:
    """
    اكتشاف ملفات Selected_Features_Kxxx.xlsx لكل طريقة.
    """
    scenario_dir = feature_selection_dir / scenario
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Missing scenario folder: {scenario_dir}")

    discovered: dict[str, dict[int, Path]] = {}

    for method_dir in sorted(path for path in scenario_dir.iterdir() if path.is_dir()):
        method = method_dir.name
        k_files: dict[int, Path] = {}

        for file_path in method_dir.glob("Selected_Features_K*.xlsx"):
            digits = "".join(character for character in file_path.stem if character.isdigit())
            if not digits:
                continue
            k = int(digits)

            # AllMsr يجب أن يرى جميع قيم K المتاحة حتى في FAST_MODE،
            # لأننا سنحدد منه عدد الخصائص الكامل ديناميكياً بأكبر K موجود.
            if method != "AllMsr" and FAST_MODE and k not in FAST_K_VALUES:
                continue

            k_files[k] = file_path

        # AllMsr هو baseline بجميع الخصائص فقط.
        # بدل القيمة الثابتة 225، نعتبر أكبر K موجود داخل مجلد AllMsr
        # هو عدد جميع الخصائص الفعلي لذلك الحجم/السيناريو.
        # أمثلة مؤكدة: 256->225، 128->900، 64->3600.
        # وعند اكتمال 32 سيُلتقط K=14400 تلقائياً إذا كان هو الأكبر.
        if method == "AllMsr" and k_files:
            max_k = max(k_files)
            k_files = {max_k: k_files[max_k]}

        if k_files:
            discovered[method] = dict(sorted(k_files.items()))

    if not discovered:
        raise ValueError(f"No feature-selection lists found in {scenario_dir}")

    return discovered


def read_selected_features(path: Path) -> list[str]:
    frame = pd.read_excel(path)
    if "Feature" not in frame.columns:
        raise KeyError(f"'Feature' column not found in: {path}")
    features = frame["Feature"].dropna().astype(str).tolist()
    if not features:
        raise ValueError(f"Empty feature list: {path}")
    return features


def detect_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) not in METADATA_COLUMNS
        and not str(column).startswith("Unnamed:")
    ]


def validate_features_exist(
    frame: pd.DataFrame,
    selected_features: list[str],
    context: str,
) -> None:
    missing = sorted(set(selected_features).difference(frame.columns))
    if missing:
        raise KeyError(
            f"{context}: {len(missing)} selected features are missing. "
            f"Examples: {missing[:10]}"
        )


# =========================================================
# النماذج
# =========================================================

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
        return GaussianNB(
            var_smoothing=params["var_smoothing"],
        )

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
            (
                "gnb",
                GaussianNB(var_smoothing=1e-9),
            ),
        ]
        return VotingClassifier(
            estimators=estimators,
            voting=params["voting"],
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# المقاييس
# =========================================================

def get_score_matrix(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X), dtype=float)
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            scores = np.asarray(model.decision_function(X), dtype=float)
            return scores
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

        y_binary = label_binarize(y_true, classes=classes)

        if score_matrix.ndim == 1:
            return float("nan")

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


def evaluate_model(
    model,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = model.predict(X_validation)
    scores = get_score_matrix(model, X_validation)

    metrics = {
        "Validation_Accuracy": accuracy_score(y_validation, predictions),
        "Validation_Precision_Macro": precision_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "Validation_Recall_Macro": recall_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "Validation_F1_Macro": f1_score(
            y_validation,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "Validation_Precision_Weighted": precision_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Validation_Recall_Weighted": recall_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Validation_F1_Weighted": f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "Validation_ROC_AUC_OVR_Macro": calculate_auc(
            y_validation,
            scores,
            classes,
        ),
    }

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=classes,
    )

    return metrics, matrix


# =========================================================
# Checkpoint
# =========================================================

RESULT_COLUMNS = [
    "Run_ID",
    "Stage",
    "Scenario",
    "Feature_Method",
    "K",
    "Model",
    "Source_Variant",
    "Parameters_JSON",
    "Train_Rows",
    "Validation_Rows",
    "Feature_Count",
    "Fit_Seconds",
    "Predict_Seconds",
    "Validation_Accuracy",
    "Validation_Precision_Macro",
    "Validation_Recall_Macro",
    "Validation_F1_Macro",
    "Validation_Precision_Weighted",
    "Validation_Recall_Weighted",
    "Validation_F1_Weighted",
    "Validation_ROC_AUC_OVR_Macro",
    "Status",
    "Error",
]


def make_run_id(
    stage: str,
    scenario: str,
    feature_method: str,
    k: int,
    model_name: str,
    params: dict[str, Any],
) -> str:
    params_text = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return "|".join(
        [
            stage,
            scenario,
            feature_method,
            str(k),
            model_name,
            params_text,
        ]
    )


def load_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
        for column in RESULT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[RESULT_COLUMNS]
    return pd.DataFrame(columns=RESULT_COLUMNS)


def append_checkpoint(
    checkpoint: pd.DataFrame,
    row: dict[str, Any],
    path: Path,
) -> pd.DataFrame:
    updated = pd.concat(
        [checkpoint, pd.DataFrame([row], columns=RESULT_COLUMNS)],
        ignore_index=True,
    )
    updated.to_csv(path, index=False)
    return updated


# =========================================================
# تشغيل تجربة واحدة
# =========================================================

def run_experiment(
    stage: str,
    scenario: str,
    feature_method: str,
    k: int,
    selected_features: list[str],
    model_name: str,
    params: dict[str, Any],
    datasets_cache: dict[tuple[str, str, str], pd.DataFrame],
    preprocessing_dir: Path,
    checkpoint: pd.DataFrame,
    checkpoint_path: Path,
    model_output_dir: Path,
) -> pd.DataFrame:
    run_id = make_run_id(
        stage,
        scenario,
        feature_method,
        k,
        model_name,
        params,
    )

    completed_ids = set(
        checkpoint.loc[checkpoint["Status"] == "Completed", "Run_ID"].astype(str)
    )
    if run_id in completed_ids:
        print("    Skip completed:", run_id)
        return checkpoint

    # حذف السجل الفاشل السابق للتجربة نفسها كي يعاد تنفيذها دون تكرار دائم.
    checkpoint = checkpoint[checkpoint["Run_ID"].astype(str) != run_id].copy()

    variant = MODEL_SOURCE_VARIANT[model_name]

    cache_key_train = (scenario, variant, "Train")
    cache_key_validation = (scenario, variant, "Validation")

    if cache_key_train not in datasets_cache:
        datasets_cache[cache_key_train] = load_variant_split(
            preprocessing_dir,
            scenario,
            variant,
            "Train",
        )

    if cache_key_validation not in datasets_cache:
        datasets_cache[cache_key_validation] = load_variant_split(
            preprocessing_dir,
            scenario,
            variant,
            "Validation",
        )

    train_frame = datasets_cache[cache_key_train]
    validation_frame = datasets_cache[cache_key_validation]

    validate_features_exist(
        train_frame,
        selected_features,
        f"{scenario}/{feature_method}/K{k}/{model_name}/Train",
    )
    validate_features_exist(
        validation_frame,
        selected_features,
        f"{scenario}/{feature_method}/K{k}/{model_name}/Validation",
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_frame[TARGET_COLUMN])
    y_validation = label_encoder.transform(validation_frame[TARGET_COLUMN])

    X_train = train_frame[selected_features].to_numpy(dtype=np.float32)
    X_validation = validation_frame[selected_features].to_numpy(dtype=np.float32)

    row: dict[str, Any] = {
        "Run_ID": run_id,
        "Stage": stage,
        "Scenario": scenario,
        "Feature_Method": feature_method,
        "K": k,
        "Model": model_name,
        "Source_Variant": variant,
        "Parameters_JSON": json.dumps(
            params,
            ensure_ascii=False,
            sort_keys=True,
        ),
        "Train_Rows": len(train_frame),
        "Validation_Rows": len(validation_frame),
        "Feature_Count": len(selected_features),
        "Status": "Failed",
        "Error": "",
    }

    try:
        model = create_model(model_name, params)

        fit_start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        metrics, matrix = evaluate_model(
            model,
            X_validation,
            y_validation,
            np.arange(len(label_encoder.classes_)),
        )
        predict_seconds = time.perf_counter() - predict_start

        row.update(metrics)
        row["Fit_Seconds"] = fit_seconds
        row["Predict_Seconds"] = predict_seconds
        row["Status"] = "Completed"

        run_folder = (
            model_output_dir
            / scenario
            / stage
            / model_name
            / feature_method
            / f"K{k:03d}"
        )
        run_folder.mkdir(parents=True, exist_ok=True)

        safe_param_id = abs(hash(row["Parameters_JSON"])) % 10**12

        pd.DataFrame(
            matrix,
            index=[f"True_{item}" for item in label_encoder.classes_],
            columns=[f"Pred_{item}" for item in label_encoder.classes_],
        ).to_excel(
            run_folder / f"ConfusionMatrix_{safe_param_id}.xlsx"
        )

        # حفظ النموذج فقط في مرحلة Tuning لتقليل حجم التخزين.
        if stage == "Tuning":
            joblib.dump(
                {
                    "model": model,
                    "label_encoder": label_encoder,
                    "selected_features": selected_features,
                    "scenario": scenario,
                    "feature_method": feature_method,
                    "k": k,
                    "model_name": model_name,
                    "parameters": params,
                    "source_variant": variant,
                },
                run_folder / f"Model_{safe_param_id}.joblib",
                compress=3,
            )

    except Exception as error:
        row["Error"] = repr(error)
        print("    FAILED:", error)

    checkpoint = append_checkpoint(
        checkpoint,
        row,
        checkpoint_path,
    )

    del X_train, X_validation, y_train, y_validation
    gc.collect()

    return checkpoint


# =========================================================
# التقارير النهائية
# =========================================================

def select_top_feature_configs(
    screening_results: pd.DataFrame,
    scenario: str,
    model_name: str,
    top_n: int,
) -> pd.DataFrame:
    subset = screening_results[
        (screening_results["Scenario"] == scenario)
        & (screening_results["Model"] == model_name)
        & (screening_results["Status"] == "Completed")
    ].copy()

    subset = subset.sort_values(
        [PRIMARY_METRIC, SECONDARY_METRIC],
        ascending=[False, False],
    )

    return subset.head(top_n)


def create_final_reports(
    checkpoint: pd.DataFrame,
    output_dir: Path,
) -> None:
    completed = checkpoint[checkpoint["Status"] == "Completed"].copy()

    if completed.empty:
        raise ValueError("No completed experiments were found.")

    # All_Validation_Results يبقى شاملاً Screening + Tuning لغرض التوثيق الكامل.
    completed = completed.sort_values(
        [
            "Scenario",
            PRIMARY_METRIC,
            SECONDARY_METRIC,
        ],
        ascending=[True, False, False],
    )

    completed.to_excel(
        output_dir / "All_Validation_Results.xlsx",
        index=False,
    )

    # التقارير التي تمثل أفضل إعداد نهائي تعتمد على Tuning فقط.
    tuning_completed = completed[
        completed["Stage"].astype(str).str.strip().eq("Tuning")
    ].copy()

    if tuning_completed.empty:
        raise ValueError(
            "No completed Tuning experiments were found. "
            "Final best-configuration reports cannot be created."
        )

    tuning_completed = tuning_completed.sort_values(
        [
            "Scenario",
            PRIMARY_METRIC,
            SECONDARY_METRIC,
        ],
        ascending=[True, False, False],
    )

    best_per_scenario = (
        tuning_completed.sort_values(
            [PRIMARY_METRIC, SECONDARY_METRIC],
            ascending=[False, False],
        )
        .groupby("Scenario", as_index=False)
        .head(1)
    )

    best_per_model = (
        tuning_completed.sort_values(
            [PRIMARY_METRIC, SECONDARY_METRIC],
            ascending=[False, False],
        )
        .groupby(["Scenario", "Model"], as_index=False)
        .head(1)
    )

    best_per_method = (
        tuning_completed.sort_values(
            [PRIMARY_METRIC, SECONDARY_METRIC],
            ascending=[False, False],
        )
        .groupby(["Scenario", "Feature_Method"], as_index=False)
        .head(1)
    )

    best_per_scenario.to_excel(
        output_dir / "Best_Configuration_Per_Scenario.xlsx",
        index=False,
    )
    best_per_model.to_excel(
        output_dir / "Best_Configuration_Per_Model.xlsx",
        index=False,
    )
    best_per_method.to_excel(
        output_dir / "Best_Configuration_Per_Feature_Method.xlsx",
        index=False,
    )

    # Validation_Results_Summary يبقى شاملاً Screening + Tuning.
    summary = (
        completed.groupby(
            ["Scenario", "Stage", "Model", "Feature_Method"],
            as_index=False,
        )
        .agg(
            Experiments=("Run_ID", "count"),
            Best_Accuracy=("Validation_Accuracy", "max"),
            Mean_Accuracy=("Validation_Accuracy", "mean"),
            Best_F1_Macro=("Validation_F1_Macro", "max"),
            Mean_F1_Macro=("Validation_F1_Macro", "mean"),
            Best_AUC=("Validation_ROC_AUC_OVR_Macro", "max"),
            Total_Fit_Seconds=("Fit_Seconds", "sum"),
        )
    )

    summary.to_excel(
        output_dir / "Validation_Results_Summary.xlsx",
        index=False,
    )



def write_model_validation_metadata(
    preprocessing_dir: Path,
    feature_selection_dir: Path,
    output_dir: Path,
    checkpoint: pd.DataFrame,
    top_n: int,
) -> None:
    """حفظ معلومات المرحلة السادسة دون إعادة أي تدريب."""
    completed = checkpoint[checkpoint["Status"] == "Completed"].copy()
    failed = checkpoint[checkpoint["Status"] != "Completed"].copy()

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessing_dir": str(preprocessing_dir),
        "feature_selection_dir": str(feature_selection_dir),
        "output_dir": str(output_dir),
        "scenarios": list(SCENARIOS),
        "models": list(ENABLED_MODELS),
        "valid_splits_loaded_by_pre6": list(VALID_SPLITS),
        "fast_mode": FAST_MODE,
        "fast_k_values": sorted(FAST_K_VALUES),
        "top_n_feature_configs_per_model": int(top_n),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "random_state": RANDOM_STATE,
        "screening_parameters": SCREENING_PARAMETERS,
        "tuning_grids": TUNING_GRIDS,
        "model_source_variant": MODEL_SOURCE_VARIANT,
        "checkpoint_file": str(output_dir / "Validation_Checkpoint.csv"),
        "completed_experiments": int(len(completed)),
        "failed_or_incomplete_experiments": int(len(failed)),
        "completed_screening_experiments": int(
            (completed["Stage"] == "Screening").sum()
        ),
        "completed_tuning_experiments": int(
            (completed["Stage"] == "Tuning").sum()
        ),
        "data_leakage_policy": (
            "Models are fitted on Train only. Configuration selection uses "
            "Validation only. Test is not loaded or used by Pre6."
        ),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    (output_dir / "Model_Validation_Metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# =========================================================
# التشغيل المجزأ بحد زمني آمن
# =========================================================

def _completed_run_ids(checkpoint: pd.DataFrame) -> set[str]:
    if checkpoint.empty:
        return set()
    return set(
        checkpoint.loc[checkpoint["Status"] == "Completed", "Run_ID"].astype(str)
    )


def _time_limit_reached(start_time: float, max_hours: float) -> bool:
    if max_hours <= 0:
        return False
    return (time.perf_counter() - start_time) >= max_hours * 3600.0


def _validate_names(values: Iterable[str], allowed: Iterable[str], kind: str) -> tuple[str, ...]:
    allowed_tuple = tuple(allowed)
    cleaned = tuple(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))
    invalid = sorted(set(cleaned).difference(allowed_tuple))
    if invalid:
        raise ValueError(f"Unknown {kind}: {invalid}. Allowed: {allowed_tuple}")
    if not cleaned:
        raise ValueError(f"At least one {kind} must be selected.")
    return cleaned


def _screening_expected_ids(
    feature_selection_dir: Path,
    scenario: str,
    model_name: str,
) -> set[str]:
    expected: set[str] = set()
    discovered = discover_feature_methods(feature_selection_dir, scenario)
    for feature_method, k_files in discovered.items():
        for k in k_files:
            expected.add(
                make_run_id(
                    "Screening",
                    scenario,
                    feature_method,
                    k,
                    model_name,
                    SCREENING_PARAMETERS[model_name],
                )
            )
    return expected


def _write_progress_metadata(
    output_dir: Path,
    part_name: str,
    stage: str,
    scenarios: tuple[str, ...],
    models: tuple[str, ...],
    max_hours: float,
    experiments_this_run: int,
    start_time: float,
    stopped_by_time_limit: bool,
) -> None:
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "part_name": part_name,
        "stage": stage,
        "scenarios": list(scenarios),
        "models": list(models),
        "max_hours_per_launch": max_hours,
        "experiments_completed_this_launch": experiments_this_run,
        "elapsed_seconds_this_launch": time.perf_counter() - start_time,
        "stopped_by_time_limit": stopped_by_time_limit,
        "fast_mode": FAST_MODE,
        "top_n_feature_configs_per_model": TOP_N_FEATURE_CONFIGS_PER_MODEL,
        "data_leakage_policy": (
            "Train is used for fitting; Validation is used for configuration selection; "
            "Test is never loaded by Pre6."
        ),
        "hardware_note": (
            "The time limit is checked between experiments. A single long SVM experiment "
            "cannot be safely interrupted without losing that experiment."
        ),
    }
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in part_name)
    (output_dir / f"Progress_{safe_name}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(
    default_stage: str = "Screening",
    default_models: tuple[str, ...] = ENABLED_MODELS,
    default_scenarios: tuple[str, ...] = SCENARIOS,
    default_part_name: str = "Pre6_Custom_Part",
    default_max_hours: float = 9.0,
) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Time-limited Pre6 runner. It stops between experiments and resumes from "
            "Validation_Checkpoint.csv on the next launch."
        )
    )
    parser.add_argument("--preprocessing-dir", type=Path, default=DEFAULT_PREPROCESSING_DIR)
    parser.add_argument("--feature-selection-dir", type=Path, default=DEFAULT_FEATURE_SELECTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N_FEATURE_CONFIGS_PER_MODEL)
    parser.add_argument("--stage", choices=("Screening", "Tuning", "Reports"), default=default_stage)
    parser.add_argument("--models", nargs="+", default=list(default_models))
    parser.add_argument("--scenarios", nargs="+", default=list(default_scenarios))
    parser.add_argument("--max-hours", type=float, default=default_max_hours)
    parser.add_argument("--part-name", type=str, default=default_part_name)

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("Ignored unknown arguments:", unknown_args)

    preprocessing_dir = Path(args.preprocessing_dir)
    feature_selection_dir = Path(args.feature_selection_dir)
    output_dir = Path(args.output_dir)
    models = _validate_names(args.models, ENABLED_MODELS, "model")
    scenarios = _validate_names(args.scenarios, SCENARIOS, "scenario")

    if args.top_n <= 0:
        raise ValueError("--top-n must be greater than zero.")
    if args.max_hours < 0:
        raise ValueError("--max-hours cannot be negative.")
    if not preprocessing_dir.exists():
        raise FileNotFoundError(preprocessing_dir)
    if not feature_selection_dir.exists():
        raise FileNotFoundError(feature_selection_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "Validation_Checkpoint.csv"
    checkpoint = load_checkpoint(checkpoint_path)

    print("=" * 80)
    print("Part                    :", args.part_name)
    print("Stage                   :", args.stage)
    print("Scenarios               :", scenarios)
    print("Models                  :", models)
    print("Maximum launch time     :", args.max_hours, "hours")
    print("Checkpoint              :", checkpoint_path)
    print("Important               : Test is NOT loaded or used.")
    print("Power-loss protection   : checkpoint is saved after every experiment.")
    print("=" * 80)

    if args.stage == "Reports":
        create_final_reports(checkpoint=checkpoint, output_dir=output_dir)
        write_model_validation_metadata(
            preprocessing_dir=preprocessing_dir,
            feature_selection_dir=feature_selection_dir,
            output_dir=output_dir,
            checkpoint=checkpoint,
            top_n=args.top_n,
        )
        print("Final reports created successfully.")
        print("Metadata created:", output_dir / "Model_Validation_Metadata.json")
        return

    datasets_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    selected_features_cache: dict[tuple[str, str, int], list[str]] = {}
    start_time = time.perf_counter()
    experiments_this_run = 0
    stopped_by_time_limit = False

    if args.stage == "Screening":
        for scenario in scenarios:
            discovered = discover_feature_methods(feature_selection_dir, scenario)
            for feature_method, k_files in discovered.items():
                for k, feature_file in k_files.items():
                    cache_key = (scenario, feature_method, k)
                    if cache_key not in selected_features_cache:
                        selected_features_cache[cache_key] = read_selected_features(feature_file)
                    selected_features = selected_features_cache[cache_key]

                    for model_name in models:
                        run_id = make_run_id(
                            "Screening", scenario, feature_method, k,
                            model_name, SCREENING_PARAMETERS[model_name]
                        )
                        if run_id in _completed_run_ids(checkpoint):
                            continue
                        if _time_limit_reached(start_time, args.max_hours):
                            stopped_by_time_limit = True
                            break

                        print(
                            f"{scenario} | {feature_method} | K={k} | "
                            f"{model_name} | Screening"
                        )
                        checkpoint = run_experiment(
                            stage="Screening",
                            scenario=scenario,
                            feature_method=feature_method,
                            k=k,
                            selected_features=selected_features,
                            model_name=model_name,
                            params=SCREENING_PARAMETERS[model_name],
                            datasets_cache=datasets_cache,
                            preprocessing_dir=preprocessing_dir,
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            model_output_dir=output_dir / "Runs",
                        )
                        experiments_this_run += 1
                    if stopped_by_time_limit:
                        break
                if stopped_by_time_limit:
                    break
            if stopped_by_time_limit:
                break

    elif args.stage == "Tuning":
        screening_completed = checkpoint[
            (checkpoint["Stage"] == "Screening")
            & (checkpoint["Status"] == "Completed")
        ].copy()
        completed_ids = _completed_run_ids(checkpoint)

        # لا يبدأ الضبط قبل اكتمال Screening للنموذج والسيناريو نفسيهما.
        incomplete_pairs: list[str] = []
        for scenario in scenarios:
            for model_name in models:
                expected = _screening_expected_ids(feature_selection_dir, scenario, model_name)
                missing_count = len(expected.difference(completed_ids))
                if missing_count:
                    incomplete_pairs.append(f"{scenario}/{model_name}: {missing_count} missing")
        if incomplete_pairs:
            raise RuntimeError(
                "Tuning cannot start before Screening is complete for these pairs:\n- "
                + "\n- ".join(incomplete_pairs)
            )

        for scenario in scenarios:
            discovered = discover_feature_methods(feature_selection_dir, scenario)
            for model_name in models:
                top_configs = select_top_feature_configs(
                    screening_results=screening_completed,
                    scenario=scenario,
                    model_name=model_name,
                    top_n=args.top_n,
                )
                if top_configs.empty:
                    raise RuntimeError(f"No completed Screening results for {scenario}/{model_name}")

                for _, config_row in top_configs.iterrows():
                    feature_method = str(config_row["Feature_Method"])
                    k = int(config_row["K"])
                    feature_file = discovered[feature_method][k]
                    cache_key = (scenario, feature_method, k)
                    if cache_key not in selected_features_cache:
                        selected_features_cache[cache_key] = read_selected_features(feature_file)
                    selected_features = selected_features_cache[cache_key]

                    for params in TUNING_GRIDS[model_name]:
                        run_id = make_run_id(
                            "Tuning", scenario, feature_method, k, model_name, params
                        )
                        if run_id in _completed_run_ids(checkpoint):
                            continue
                        if _time_limit_reached(start_time, args.max_hours):
                            stopped_by_time_limit = True
                            break

                        print(
                            f"{scenario} | {feature_method} | K={k} | "
                            f"{model_name} | Tuning | {params}"
                        )
                        checkpoint = run_experiment(
                            stage="Tuning",
                            scenario=scenario,
                            feature_method=feature_method,
                            k=k,
                            selected_features=selected_features,
                            model_name=model_name,
                            params=params,
                            datasets_cache=datasets_cache,
                            preprocessing_dir=preprocessing_dir,
                            checkpoint=checkpoint,
                            checkpoint_path=checkpoint_path,
                            model_output_dir=output_dir / "Runs",
                        )
                        experiments_this_run += 1
                    if stopped_by_time_limit:
                        break
                if stopped_by_time_limit:
                    break
            if stopped_by_time_limit:
                break

    _write_progress_metadata(
        output_dir=output_dir,
        part_name=args.part_name,
        stage=args.stage,
        scenarios=scenarios,
        models=models,
        max_hours=args.max_hours,
        experiments_this_run=experiments_this_run,
        start_time=start_time,
        stopped_by_time_limit=stopped_by_time_limit,
    )

    elapsed_hours = (time.perf_counter() - start_time) / 3600.0
    print("=" * 80)
    print("Experiments completed this launch:", experiments_this_run)
    print(f"Elapsed time: {elapsed_hours:.3f} hours")
    if stopped_by_time_limit:
        print("TIME LIMIT REACHED SAFELY BETWEEN EXPERIMENTS.")
        print("Run this SAME part again; it will resume from the checkpoint.")
    else:
        print("THIS PART IS COMPLETE. Proceed to the next numbered part.")
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
