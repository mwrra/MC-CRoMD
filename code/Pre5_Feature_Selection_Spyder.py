# -*- coding: utf-8 -*-
"""
Pre5_Feature_Selection_Spyder.py

اختيار وترتيب خصائص MC-CRoMD بعد تنفيذ Pre4_Data_Preprocessing.

السيناريوهات:
- Binary
- Three_Class
- Five_Class

الطرق:
1) AllMsr   : جميع الخصائص، خط أساس Baseline
2) Variance : ترتيب الخصائص حسب تباين Train
3) Chi2     : اختبار Chi-Square على MinMaxScaled
4) Fclass   : ANOVA F-value على StandardScaled
5) Mutclass : Mutual Information على StandardScaled
6) Random   : Random Forest Feature Importance على Clean_Unscaled
7) Logistic : L1 Logistic Regression على StandardScaled

المبدأ العلمي:
- جميع طرق الاختيار تُدرّب على Train فقط.
- Validation وTest لا يدخلان في حساب الدرجات أو الترتيب.
- البرنامج يولد ترتيبًا كاملًا وقوائم مرشحة لعدة قيم K.
- اختيار أفضل K يتم لاحقًا باستخدام Validation فقط.
- Test لا يُستخدم لاختيار الطريقة أو عدد الخصائص.

المخرجات:
- تقرير ترتيب لكل طريقة.
- قوائم الخصائص المختارة لكل K.
- ملف Summary شامل.
- ملفات Pickle للدرجات والترتيبات.
- يمكن اختياريًا إنشاء ملفات Train/Validation/Test المختصرة لكل K.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning


# =========================================================
# الإعدادات الافتراضية
# =========================================================

DEFAULT_PREPROCESSING_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\3-DataPreprocessing\64"
)

DEFAULT_OUTPUT_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\4-FeatureSelection\64"
)

# أعداد الخصائص المرشحة. سيُختار الأفضل لاحقًا باستخدام Validation.
CANDIDATE_K_VALUES = (25, 50, 75, 100, 125, 150, 175, 200, 225)

RANDOM_STATE = 42

# لتجنب إنشاء مئات الملفات الكبيرة، الوضع الافتراضي يحفظ القوائم فقط.
# غيّرها إلى True لإنشاء ملفات بيانات مختصرة لكل K.
SAVE_SELECTED_DATASETS = False

# صيغة ملفات البيانات المختصرة عند تفعيل الخيار أعلاه.
SELECTED_DATA_FORMAT = "csv"  # csv أو xlsx

# إعداد Random Forest.
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1

# إعداد Logistic Regression.
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 5000
LOGISTIC_TOL = 1e-4

SCENARIOS = ("Binary", "Three_Class", "Five_Class")
VALID_SPLITS = ("Train", "Validation", "Test")

METHOD_SOURCE_VARIANT = {
    "AllMsr": "Clean_Unscaled",
    "Variance": "Clean_Unscaled",
    "Chi2": "MinMaxScaled",
    "Fclass": "StandardScaled",
    "Mutclass": "StandardScaled",
    "Random": "Clean_Unscaled",
    "Logistic": "StandardScaled",
}

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


def find_existing_file(folder: Path, stem: str) -> Path:
    """العثور على CSV.GZ أو CSV أو XLSX بالاسم نفسه."""
    candidates = (
        folder / f"{stem}.csv.gz",
        folder / f"{stem}.csv",
        folder / f"{stem}.xlsx",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find any of:\n" + "\n".join(str(path) for path in candidates)
    )


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {path}")


def save_table(frame: pd.DataFrame, base_path: Path, output_format: str) -> Path:
    if output_format == "csv":
        output_path = base_path.with_suffix(".csv.gz")
        frame.to_csv(output_path, index=False, compression="gzip")
    elif output_format == "xlsx":
        output_path = base_path.with_suffix(".xlsx")
        frame.to_excel(output_path, index=False)
    else:
        raise ValueError("output_format must be csv or xlsx")
    return output_path


def detect_feature_columns(frame: pd.DataFrame) -> list[str]:
    features = [
        str(column)
        for column in frame.columns
        if str(column) not in METADATA_COLUMNS
        and not str(column).startswith("Unnamed:")
    ]
    if not features:
        raise ValueError("No feature columns detected.")
    return features


def load_scenario_variant(
    preprocessing_dir: Path,
    scenario: str,
    variant: str,
) -> dict[str, pd.DataFrame]:
    """تحميل Train وValidation وTest لنسخة معالجة معينة."""
    variant_dir = preprocessing_dir / scenario / variant
    loaded: dict[str, pd.DataFrame] = {}

    for split_name in VALID_SPLITS:
        file_path = find_existing_file(variant_dir, split_name)
        frame = read_table(file_path)
        frame.columns = [str(column).strip() for column in frame.columns]
        loaded[split_name] = frame

    return loaded


def validate_split_alignment(
    datasets_by_variant: dict[str, dict[str, pd.DataFrame]],
) -> None:
    """
    التأكد من أن نسخ Clean/Standard/MinMax لها الصفوف نفسها والترتيب نفسه.
    """
    variants = list(datasets_by_variant)
    reference_variant = variants[0]

    id_candidates = ("ImageSHA256", "FileName", "ImgName")

    for split_name in VALID_SPLITS:
        reference = datasets_by_variant[reference_variant][split_name]

        id_column = next(
            (column for column in id_candidates if column in reference.columns),
            None,
        )

        for variant in variants[1:]:
            current = datasets_by_variant[variant][split_name]

            if len(current) != len(reference):
                raise ValueError(
                    f"{split_name}: row count mismatch between "
                    f"{reference_variant} and {variant}"
                )

            if id_column and id_column in current.columns:
                if not reference[id_column].astype(str).reset_index(drop=True).equals(
                    current[id_column].astype(str).reset_index(drop=True)
                ):
                    raise ValueError(
                        f"{split_name}: row order mismatch in {id_column} "
                        f"between {reference_variant} and {variant}"
                    )


def sanitize_scores(scores: np.ndarray) -> np.ndarray:
    """تحويل NaN و±inf إلى قيم قابلة للترتيب."""
    scores = np.asarray(scores, dtype=float)
    return np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(float).max,
        neginf=-np.inf,
    )


def rank_features(
    feature_names: list[str],
    scores: np.ndarray,
    method: str,
    extra: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    clean_scores = sanitize_scores(scores)
    order = np.argsort(-clean_scores, kind="mergesort")

    ranking = pd.DataFrame(
        {
            "Method": method,
            "Rank": np.arange(1, len(feature_names) + 1),
            "Feature": np.asarray(feature_names, dtype=object)[order],
            "Score": clean_scores[order],
        }
    )

    if extra:
        for column_name, values in extra.items():
            values = np.asarray(values)
            ranking[column_name] = values[order]

    return ranking


def compute_method_ranking(
    method: str,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X = train_frame[feature_columns].to_numpy(dtype=float)
    y = train_frame[target_column].to_numpy()

    fitted_info: dict[str, Any] = {
        "method": method,
        "feature_names": feature_columns,
    }

    if method == "AllMsr":
        # إبقاء ترتيب الأعمدة الأصلي.
        ranking = pd.DataFrame(
            {
                "Method": method,
                "Rank": np.arange(1, len(feature_columns) + 1),
                "Feature": feature_columns,
                "Score": np.ones(len(feature_columns), dtype=float),
            }
        )
        fitted_info["description"] = "All features baseline"

    elif method == "Variance":
        scores = np.var(X, axis=0, ddof=0)
        ranking = rank_features(feature_columns, scores, method)
        fitted_info["variance_scores"] = scores

    elif method == "Chi2":
        minimum = np.nanmin(X)
        if minimum < -1e-12:
            raise ValueError(
                f"Chi2 requires nonnegative values, but minimum={minimum}"
            )
        scores, p_values = chi2(X, y)
        ranking = rank_features(
            feature_columns,
            scores,
            method,
            extra={"P_Value": p_values},
        )
        fitted_info["scores"] = scores
        fitted_info["p_values"] = p_values

    elif method == "Fclass":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores, p_values = f_classif(X, y)

        ranking = rank_features(
            feature_columns,
            scores,
            method,
            extra={"P_Value": p_values},
        )
        fitted_info["scores"] = scores
        fitted_info["p_values"] = p_values

    elif method == "Mutclass":
        scores = mutual_info_classif(
            X,
            y,
            discrete_features=False,
            random_state=RANDOM_STATE,
            n_neighbors=3,
        )
        ranking = rank_features(feature_columns, scores, method)
        fitted_info["scores"] = scores

    elif method == "Random":
        model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X, y)
        scores = model.feature_importances_

        ranking = rank_features(feature_columns, scores, method)
        fitted_info["model"] = model
        fitted_info["scores"] = scores

    elif method == "Logistic":
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=LOGISTIC_C,
            max_iter=LOGISTIC_MAX_ITER,
            tol=LOGISTIC_TOL,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=ConvergenceWarning)
            model.fit(X, y)

        coefficients = np.asarray(model.coef_, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)

        absolute_coefficients = np.abs(coefficients)
        scores = absolute_coefficients.mean(axis=0)
        nonzero_class_count = (absolute_coefficients > 1e-12).sum(axis=0)

        ranking = rank_features(
            feature_columns,
            scores,
            method,
            extra={"Nonzero_Class_Count": nonzero_class_count},
        )

        fitted_info["model"] = model
        fitted_info["scores"] = scores
        fitted_info["convergence_warnings"] = [
            str(item.message) for item in caught
        ]

    else:
        raise ValueError(f"Unknown method: {method}")

    ranking["Selected_Nonzero"] = ranking["Score"] > 0
    return ranking, fitted_info


def valid_k_values(feature_count: int) -> list[int]:
    values = sorted(
        {
            min(int(k), feature_count)
            for k in CANDIDATE_K_VALUES
            if int(k) > 0
        }
    )
    if feature_count not in values:
        values.append(feature_count)
    return values


def save_feature_lists(
    method_dir: Path,
    ranking: pd.DataFrame,
    scenario: str,
    method: str,
    k_values: list[int],
) -> dict[int, list[str]]:
    selected_by_k: dict[int, list[str]] = {}

    ranking.to_excel(
        method_dir / "Feature_Ranking.xlsx",
        index=False,
    )
    ranking.to_csv(
        method_dir / "Feature_Ranking.csv",
        index=False,
    )

    list_rows: list[dict[str, Any]] = []

    for k in k_values:
        selected = ranking.head(k)["Feature"].tolist()
        selected_by_k[k] = selected

        pd.DataFrame(
            {
                "Scenario": scenario,
                "Method": method,
                "K": k,
                "Rank": np.arange(1, k + 1),
                "Feature": selected,
            }
        ).to_excel(
            method_dir / f"Selected_Features_K{k:03d}.xlsx",
            index=False,
        )

        for rank, feature in enumerate(selected, start=1):
            list_rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "K": k,
                    "Rank": rank,
                    "Feature": feature,
                }
            )

    pd.DataFrame(list_rows).to_csv(
        method_dir / "All_Selected_Feature_Lists.csv",
        index=False,
    )

    (method_dir / "Selected_Features.json").write_text(
        json.dumps(
            {str(k): features for k, features in selected_by_k.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return selected_by_k


def save_selected_datasets(
    datasets: dict[str, pd.DataFrame],
    selected_by_k: dict[int, list[str]],
    method_dir: Path,
    output_format: str,
) -> list[Path]:
    saved: list[Path] = []

    for k, features in selected_by_k.items():
        k_dir = method_dir / f"K{k:03d}"
        k_dir.mkdir(parents=True, exist_ok=True)

        for split_name, frame in datasets.items():
            metadata_columns = [
                column for column in frame.columns if column not in features
            ]
            # نحافظ فقط على الأعمدة التعريفية والخصائص المختارة.
            metadata_columns = [
                column
                for column in metadata_columns
                if column in METADATA_COLUMNS
            ]
            reduced = frame[metadata_columns + features].copy()

            saved.append(
                save_table(
                    reduced,
                    k_dir / split_name,
                    output_format,
                )
            )

    return saved


def jaccard_similarity(list_a: list[str], list_b: list[str]) -> float:
    set_a = set(list_a)
    set_b = set(list_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe feature selection for MC-CRoMD."
    )
    parser.add_argument(
        "--preprocessing-dir",
        type=Path,
        default=DEFAULT_PREPROCESSING_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--target-column",
        default="label",
    )
    parser.add_argument(
        "--save-selected-datasets",
        action="store_true",
        default=SAVE_SELECTED_DATASETS,
    )
    parser.add_argument(
        "--selected-data-format",
        choices=("csv", "xlsx"),
        default=SELECTED_DATA_FORMAT,
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("Ignored unknown arguments:", unknown_args)

    preprocessing_dir = Path(args.preprocessing_dir)
    output_dir = Path(args.output_dir)

    print("Preprocessing folder:", preprocessing_dir)
    print("Output folder       :", output_dir)
    print("Target column       :", args.target_column)
    print("Save selected data  :", args.save_selected_datasets)

    if not preprocessing_dir.exists():
        raise FileNotFoundError(
            f"Preprocessing folder does not exist: {preprocessing_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    global_summary: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        print("=" * 72)
        print("Scenario:", scenario)

        scenario_output = output_dir / scenario
        scenario_output.mkdir(parents=True, exist_ok=True)

        required_variants = sorted(set(METHOD_SOURCE_VARIANT.values()))
        datasets_by_variant = {
            variant: load_scenario_variant(
                preprocessing_dir,
                scenario,
                variant,
            )
            for variant in required_variants
        }

        validate_split_alignment(datasets_by_variant)

        reference_train = datasets_by_variant["Clean_Unscaled"]["Train"]

        if args.target_column not in reference_train.columns:
            raise KeyError(
                f"{scenario}: target column '{args.target_column}' not found."
            )

        feature_columns = detect_feature_columns(reference_train)
        feature_count = len(feature_columns)
        k_values = valid_k_values(feature_count)

        print("Train rows     :", len(reference_train))
        print("Feature count  :", feature_count)
        print("Candidate K    :", k_values)

        scenario_selected: dict[str, dict[int, list[str]]] = {}

        for method, source_variant in METHOD_SOURCE_VARIANT.items():
            print(f"  Method: {method} | Source: {source_variant}")

            method_start = time.perf_counter()
            method_dir = scenario_output / method
            method_dir.mkdir(parents=True, exist_ok=True)

            source_datasets = datasets_by_variant[source_variant]
            train_frame = source_datasets["Train"]

            current_features = detect_feature_columns(train_frame)
            if current_features != feature_columns:
                if set(current_features) != set(feature_columns):
                    raise ValueError(
                        f"{scenario}/{method}: feature columns differ "
                        "between preprocessing variants."
                    )
                current_features = feature_columns

            ranking, fitted_info = compute_method_ranking(
                method=method,
                train_frame=train_frame,
                feature_columns=current_features,
                target_column=args.target_column,
            )

            selected_by_k = save_feature_lists(
                method_dir=method_dir,
                ranking=ranking,
                scenario=scenario,
                method=method,
                k_values=k_values,
            )
            scenario_selected[method] = selected_by_k

            with open(method_dir / "Fitted_Feature_Selection.pkl", "wb") as file:
                pickle.dump(fitted_info, file)

            saved_dataset_count = 0
            if args.save_selected_datasets:
                saved_paths = save_selected_datasets(
                    datasets=source_datasets,
                    selected_by_k=selected_by_k,
                    method_dir=method_dir,
                    output_format=args.selected_data_format,
                )
                saved_dataset_count = len(saved_paths)

            elapsed_method = time.perf_counter() - method_start

            nonzero_count = int((ranking["Score"] > 0).sum())
            top_feature = ranking.iloc[0]["Feature"]
            top_score = ranking.iloc[0]["Score"]

            global_summary.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "Source_Variant": source_variant,
                    "Train_Rows": len(train_frame),
                    "Feature_Count": feature_count,
                    "Nonzero_Score_Features": nonzero_count,
                    "Top_Feature": top_feature,
                    "Top_Score": top_score,
                    "Candidate_K_Values": ",".join(map(str, k_values)),
                    "Saved_Selected_Dataset_Files": saved_dataset_count,
                    "Execution_Seconds": elapsed_method,
                }
            )

        # قياس تشابه القوائم بين طرق الاختيار لكل K.
        methods = list(METHOD_SOURCE_VARIANT)
        for k in k_values:
            for index_a in range(len(methods)):
                for index_b in range(index_a + 1, len(methods)):
                    method_a = methods[index_a]
                    method_b = methods[index_b]

                    list_a = scenario_selected[method_a][k]
                    list_b = scenario_selected[method_b][k]

                    overlap_rows.append(
                        {
                            "Scenario": scenario,
                            "K": k,
                            "Method_A": method_a,
                            "Method_B": method_b,
                            "Intersection_Count": len(set(list_a) & set(list_b)),
                            "Jaccard_Similarity": jaccard_similarity(
                                list_a,
                                list_b,
                            ),
                        }
                    )

    summary_frame = pd.DataFrame(global_summary)
    overlap_frame = pd.DataFrame(overlap_rows)

    summary_frame.to_excel(
        output_dir / "Feature_Selection_Summary.xlsx",
        index=False,
    )
    overlap_frame.to_excel(
        output_dir / "Feature_Selection_Method_Overlap.xlsx",
        index=False,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "preprocessing_dir": str(preprocessing_dir),
        "output_dir": str(output_dir),
        "scenarios": list(SCENARIOS),
        "methods": METHOD_SOURCE_VARIANT,
        "candidate_k_values": list(CANDIDATE_K_VALUES),
        "random_state": RANDOM_STATE,
        "save_selected_datasets": bool(args.save_selected_datasets),
        "selected_data_format": args.selected_data_format,
        "selection_policy": (
            "Feature scores and rankings fitted on Train only. "
            "Best K must be selected later using Validation only."
        ),
        "test_policy": (
            "Test must not be used for choosing method, K, model, "
            "or hyperparameters."
        ),
        "rf_parameters": {
            "n_estimators": RF_N_ESTIMATORS,
            "max_depth": RF_MAX_DEPTH,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        },
        "logistic_parameters": {
            "penalty": "l1",
            "solver": "saga",
            "C": LOGISTIC_C,
            "max_iter": LOGISTIC_MAX_ITER,
            "tol": LOGISTIC_TOL,
        },
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    (output_dir / "Feature_Selection_Metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start_time

    print("=" * 72)
    print("Feature selection completed successfully.")
    print("Summary :", output_dir / "Feature_Selection_Summary.xlsx")
    print(
        "Overlap :",
        output_dir / "Feature_Selection_Method_Overlap.xlsx",
    )
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
