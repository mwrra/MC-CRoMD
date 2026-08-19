# -*- coding: utf-8 -*-
"""
Pre4_Data_Preprocessing_Spyder.py

المعالجة المسبقة الآمنة لخصائص MC-CRoMD بعد:
1) ربط SourceGroupID
2) إنشاء Split ثابت
3) تنفيذ Pre3_Data_Quality_Audit

ينفذ لكل سيناريو:
- Binary
- Three_Class
- Five_Class

الخطوات:
1) استبعاد الأعمدة التعريفية من المعالجة الرقمية.
2) تحويل القيم غير الرقمية و ±inf إلى NaN.
3) اعتبار القيم السالبة غير المنطقية NaN، مع السماح بالسالب في Cronbach.
4) حذف الخصائص الثابتة اعتمادًا على Train فقط.
5) تعويض القيم المفقودة بوسيط Train فقط.
6) قص القيم المتطرفة بحدود IQR المحسوبة من Train فقط.
7) إنشاء ثلاث نسخ:
   - Clean_Unscaled
   - StandardScaled
   - MinMaxScaled
8) تطبيق معاملات Train نفسها على Validation وTest.
9) حفظ جميع المعاملات والخصائص المستخدمة لإعادة التجربة.

مهم:
- لا يتم استخدام Validation أو Test لحساب أي إحصائية.
- لا يتم تنفيذ Feature Selection داخل هذا البرنامج.
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =========================================================
# الإعدادات الافتراضية
# =========================================================

DEFAULT_INPUT_FILE = Path(
    r"G:\My Research About Lung Canser\INASS\1-SourceGroupID"
    r"\GroupedSplitResults\64_5Clas\Features_With_Groups_And_Split.xlsx"
)

DEFAULT_OUTPUT_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\3-DataPreprocessing\64"
)

DEFAULT_IQR_FACTOR = 1.5

# csv أسرع وأخف من Excel للملفات الكبيرة.
# القيم المسموحة: "csv" أو "xlsx"
DEFAULT_OUTPUT_FORMAT = "csv"

SCENARIOS: dict[str, tuple[str, ...]] = {
    "Binary": ("colon_aca", "colon_n"),
    "Three_Class": ("lung_aca", "lung_scc", "lung_n"),
    "Five_Class": (
        "colon_aca",
        "colon_n",
        "lung_aca",
        "lung_scc",
        "lung_n",
    ),
}

VALID_SPLITS = ("Train", "Validation", "Test")

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


def normalize_split(value: Any) -> str:
    text = str(value).strip().lower()
    mapping = {
        "train": "Train",
        "training": "Train",
        "validation": "Validation",
        "valid": "Validation",
        "val": "Validation",
        "test": "Test",
        "testing": "Test",
    }
    return mapping.get(text, str(value).strip())


def is_cronbach_feature(column_name: str) -> bool:
    return column_name.lower().startswith("cronbach")


def expected_nonnegative(column_name: str) -> bool:
    return not is_cronbach_feature(column_name)


def read_input(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def detect_feature_columns(frame: pd.DataFrame) -> list[str]:
    feature_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in METADATA_COLUMNS
        and not str(column).startswith("Unnamed:")
    ]

    if not feature_columns:
        raise ValueError("No feature columns were detected.")

    return feature_columns


def prepare_numeric_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    تحويل الخصائص إلى أرقام:
    - النصوص غير الرقمية -> NaN
    - ±inf -> NaN
    - السالب غير المنطقي -> NaN
    """
    numeric = pd.DataFrame(index=frame.index)
    counters = {
        "non_numeric_to_nan": 0,
        "positive_inf_to_nan": 0,
        "negative_inf_to_nan": 0,
        "invalid_negative_to_nan": 0,
    }

    for feature in feature_columns:
        converted = pd.to_numeric(frame[feature], errors="coerce")

        original_missing = frame[feature].isna()
        non_numeric_mask = converted.isna() & ~original_missing
        counters["non_numeric_to_nan"] += int(non_numeric_mask.sum())

        pos_inf_mask = np.isposinf(converted.to_numpy(dtype=float, na_value=np.nan))
        neg_inf_mask = np.isneginf(converted.to_numpy(dtype=float, na_value=np.nan))

        counters["positive_inf_to_nan"] += int(pos_inf_mask.sum())
        counters["negative_inf_to_nan"] += int(neg_inf_mask.sum())

        converted = converted.replace([np.inf, -np.inf], np.nan)

        if expected_nonnegative(feature):
            invalid_negative = converted < 0
            counters["invalid_negative_to_nan"] += int(invalid_negative.sum())
            converted = converted.mask(invalid_negative, np.nan)

        numeric[feature] = converted.astype(float)

    return numeric, counters


def fit_train_preprocessing(
    train_numeric: pd.DataFrame,
    iqr_factor: float,
) -> dict[str, Any]:
    """
    تدريب جميع معاملات المعالجة على Train فقط.
    """
    # حذف الخصائص التي لا تحتوي أي قيمة صالحة في Train.
    all_missing_features = [
        column
        for column in train_numeric.columns
        if train_numeric[column].notna().sum() == 0
    ]

    candidate_features = [
        column
        for column in train_numeric.columns
        if column not in all_missing_features
    ]

    # الوسيط من Train فقط.
    medians = train_numeric[candidate_features].median(axis=0)

    # التعويض المؤقت لتحديد الخصائص الثابتة.
    train_imputed = train_numeric[candidate_features].fillna(medians)

    constant_features = [
        column
        for column in candidate_features
        if train_imputed[column].nunique(dropna=True) <= 1
    ]

    retained_features = [
        column
        for column in candidate_features
        if column not in constant_features
    ]

    medians = medians[retained_features]
    train_imputed = train_numeric[retained_features].fillna(medians)

    q1 = train_imputed.quantile(0.25)
    q3 = train_imputed.quantile(0.75)
    iqr = q3 - q1

    lower_bounds = q1 - iqr_factor * iqr
    upper_bounds = q3 + iqr_factor * iqr

    # إذا كان IQR = 0 لا ننفذ قصًا على هذه الخاصية.
    zero_iqr_features = list(iqr[iqr <= 0].index)
    lower_bounds.loc[zero_iqr_features] = -np.inf
    upper_bounds.loc[zero_iqr_features] = np.inf

    train_clipped = train_imputed.clip(
        lower=lower_bounds,
        upper=upper_bounds,
        axis=1,
    )

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_scaler.fit(train_clipped)
    minmax_scaler.fit(train_clipped)

    return {
        "all_missing_features": all_missing_features,
        "constant_features": constant_features,
        "retained_features": retained_features,
        "medians": medians,
        "q1": q1[retained_features],
        "q3": q3[retained_features],
        "iqr": iqr[retained_features],
        "lower_bounds": lower_bounds[retained_features],
        "upper_bounds": upper_bounds[retained_features],
        "zero_iqr_features": zero_iqr_features,
        "standard_scaler": standard_scaler,
        "minmax_scaler": minmax_scaler,
    }


def transform_numeric(
    numeric_frame: pd.DataFrame,
    fitted: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    retained = fitted["retained_features"]

    clean = numeric_frame[retained].copy()

    missing_before = int(clean.isna().sum().sum())
    clean = clean.fillna(fitted["medians"])
    missing_after = int(clean.isna().sum().sum())

    below_mask = clean.lt(fitted["lower_bounds"], axis=1)
    above_mask = clean.gt(fitted["upper_bounds"], axis=1)

    lower_clipped_count = int(below_mask.sum().sum())
    upper_clipped_count = int(above_mask.sum().sum())

    clean = clean.clip(
        lower=fitted["lower_bounds"],
        upper=fitted["upper_bounds"],
        axis=1,
    )

    standard_values = fitted["standard_scaler"].transform(clean)
    minmax_values = fitted["minmax_scaler"].transform(clean)

    standard = pd.DataFrame(
        standard_values,
        columns=retained,
        index=clean.index,
    )
    minmax = pd.DataFrame(
        minmax_values,
        columns=retained,
        index=clean.index,
    )

    stats = {
        "missing_before_imputation": missing_before,
        "missing_after_imputation": missing_after,
        "lower_clipped_values": lower_clipped_count,
        "upper_clipped_values": upper_clipped_count,
    }

    return clean, standard, minmax, stats


def combine_metadata_and_features(
    source_frame: pd.DataFrame,
    transformed_features: pd.DataFrame,
    metadata_columns: list[str],
) -> pd.DataFrame:
    metadata = source_frame.loc[transformed_features.index, metadata_columns].copy()
    result = pd.concat(
        [
            metadata.reset_index(drop=True),
            transformed_features.reset_index(drop=True),
        ],
        axis=1,
    )
    return result


def save_frame(
    frame: pd.DataFrame,
    output_path_without_suffix: Path,
    output_format: str,
) -> Path:
    if output_format == "csv":
        output_path = output_path_without_suffix.with_suffix(".csv.gz")
        frame.to_csv(output_path, index=False, compression="gzip")
    elif output_format == "xlsx":
        output_path = output_path_without_suffix.with_suffix(".xlsx")
        frame.to_excel(output_path, index=False)
    else:
        raise ValueError("Output format must be 'csv' or 'xlsx'.")

    return output_path


def save_split_files(
    full_frame: pd.DataFrame,
    scenario_dir: Path,
    variant_name: str,
    output_format: str,
) -> list[Path]:
    variant_dir = scenario_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    # الملف الكامل للسيناريو.
    saved_paths.append(
        save_frame(
            full_frame,
            variant_dir / f"{variant_name}_All",
            output_format,
        )
    )

    # ملفات منفصلة حسب Split.
    for split_name in VALID_SPLITS:
        split_frame = full_frame[full_frame["Split"] == split_name].copy()
        saved_paths.append(
            save_frame(
                split_frame,
                variant_dir / split_name,
                output_format,
            )
        )

    return saved_paths


def save_fitted_objects(
    scenario_dir: Path,
    fitted: dict[str, Any],
) -> None:
    objects_dir = scenario_dir / "Fitted_Objects"
    objects_dir.mkdir(parents=True, exist_ok=True)

    with open(objects_dir / "Preprocessing_Objects.pkl", "wb") as file:
        pickle.dump(fitted, file)

    pd.DataFrame(
        {
            "Feature": fitted["retained_features"],
            "Median_From_Train": fitted["medians"].values,
            "Q1_From_Train": fitted["q1"].values,
            "Q3_From_Train": fitted["q3"].values,
            "IQR_From_Train": fitted["iqr"].values,
            "Lower_Bound_From_Train": fitted["lower_bounds"].values,
            "Upper_Bound_From_Train": fitted["upper_bounds"].values,
        }
    ).to_excel(
        objects_dir / "Train_Preprocessing_Parameters.xlsx",
        index=False,
    )

    pd.DataFrame(
        {"Retained_Feature": fitted["retained_features"]}
    ).to_excel(
        objects_dir / "Retained_Features.xlsx",
        index=False,
    )

    pd.DataFrame(
        {"Removed_All_Missing_Feature": fitted["all_missing_features"]}
    ).to_excel(
        objects_dir / "Removed_All_Missing_Features.xlsx",
        index=False,
    )

    pd.DataFrame(
        {"Removed_Constant_Feature": fitted["constant_features"]}
    ).to_excel(
        objects_dir / "Removed_Constant_Features.xlsx",
        index=False,
    )


def make_scenario_report(
    scenario_name: str,
    scenario_frame: pd.DataFrame,
    original_feature_count: int,
    fitted: dict[str, Any],
    conversion_counters: dict[str, int],
    transform_stats: dict[str, dict[str, int]],
    saved_paths: list[Path],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"Item": "Scenario", "Value": scenario_name},
        {"Item": "Rows", "Value": len(scenario_frame)},
        {"Item": "Original feature count", "Value": original_feature_count},
        {"Item": "Retained feature count", "Value": len(fitted["retained_features"])},
        {
            "Item": "Removed all-missing features",
            "Value": len(fitted["all_missing_features"]),
        },
        {
            "Item": "Removed constant features",
            "Value": len(fitted["constant_features"]),
        },
        {
            "Item": "Zero-IQR features not clipped",
            "Value": len(fitted["zero_iqr_features"]),
        },
    ]

    for key, value in conversion_counters.items():
        rows.append({"Item": key, "Value": value})

    for split_name, stats in transform_stats.items():
        for key, value in stats.items():
            rows.append(
                {
                    "Item": f"{split_name}: {key}",
                    "Value": value,
                }
            )

    for class_name, count in (
        scenario_frame["ClassName"].value_counts().sort_index().items()
    ):
        rows.append(
            {
                "Item": f"Class rows: {class_name}",
                "Value": int(count),
            }
        )

    for split_name in VALID_SPLITS:
        rows.append(
            {
                "Item": f"{split_name} rows",
                "Value": int((scenario_frame["Split"] == split_name).sum()),
            }
        )

    for path in saved_paths:
        rows.append({"Item": "Saved file", "Value": str(path)})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe preprocessing for MC-CRoMD features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--sheet",
        default=0,
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=DEFAULT_IQR_FACTOR,
    )
    parser.add_argument(
        "--output-format",
        choices=("csv", "xlsx"),
        default=DEFAULT_OUTPUT_FORMAT,
    )

    # مناسب للتشغيل من Spyder.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("Ignored unknown arguments:", unknown_args)

    input_file = Path(args.input)
    output_dir = Path(args.output_dir)

    print("Input file   :", input_file)
    print("Output folder:", output_dir)
    print("IQR factor   :", args.iqr_factor)
    print("Output format:", args.output_format)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    if args.iqr_factor <= 0:
        raise ValueError("IQR factor must be greater than zero.")

    start_time = time.perf_counter()

    frame = read_input(input_file, sheet_name=args.sheet)
    frame.columns = [str(column).strip() for column in frame.columns]

    required_columns = {"ClassName", "Split"}
    missing_required = required_columns.difference(frame.columns)
    if missing_required:
        raise KeyError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    frame["ClassName"] = frame["ClassName"].astype(str).str.strip()
    frame["Split"] = frame["Split"].map(normalize_split)

    unknown_splits = sorted(
        set(frame["Split"].dropna().unique()).difference(VALID_SPLITS)
    )
    if unknown_splits:
        raise ValueError(f"Unknown Split values: {unknown_splits}")

    feature_columns = detect_feature_columns(frame)
    metadata_columns = [
        column for column in frame.columns if column not in feature_columns
    ]

    print("Rows            :", len(frame))
    print("Feature columns :", len(feature_columns))

    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_frames: list[pd.DataFrame] = []

    for scenario_name, scenario_classes in SCENARIOS.items():
        print("=" * 70)
        print("Processing scenario:", scenario_name)

        scenario_frame = frame[
            frame["ClassName"].isin(scenario_classes)
        ].copy()

        missing_classes = sorted(
            set(scenario_classes).difference(
                set(scenario_frame["ClassName"].unique())
            )
        )
        if missing_classes:
            raise ValueError(
                f"{scenario_name}: missing classes: {missing_classes}"
            )

        numeric_all, conversion_counters = prepare_numeric_frame(
            scenario_frame,
            feature_columns,
        )

        train_mask = scenario_frame["Split"] == "Train"
        train_numeric = numeric_all.loc[train_mask]

        if train_numeric.empty:
            raise ValueError(f"{scenario_name}: Train split is empty.")

        fitted = fit_train_preprocessing(
            train_numeric=train_numeric,
            iqr_factor=args.iqr_factor,
        )

        scenario_dir = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        save_fitted_objects(scenario_dir, fitted)

        clean_parts: list[pd.DataFrame] = []
        standard_parts: list[pd.DataFrame] = []
        minmax_parts: list[pd.DataFrame] = []
        transform_stats: dict[str, dict[str, int]] = {}

        for split_name in VALID_SPLITS:
            split_mask = scenario_frame["Split"] == split_name
            split_numeric = numeric_all.loc[split_mask]

            clean, standard, minmax, stats = transform_numeric(
                split_numeric,
                fitted,
            )

            transform_stats[split_name] = stats
            clean_parts.append(clean)
            standard_parts.append(standard)
            minmax_parts.append(minmax)

        clean_all = pd.concat(clean_parts).sort_index()
        standard_all = pd.concat(standard_parts).sort_index()
        minmax_all = pd.concat(minmax_parts).sort_index()

        clean_output = combine_metadata_and_features(
            scenario_frame,
            clean_all,
            metadata_columns,
        )
        standard_output = combine_metadata_and_features(
            scenario_frame,
            standard_all,
            metadata_columns,
        )
        minmax_output = combine_metadata_and_features(
            scenario_frame,
            minmax_all,
            metadata_columns,
        )

        saved_paths: list[Path] = []
        saved_paths.extend(
            save_split_files(
                clean_output,
                scenario_dir,
                "Clean_Unscaled",
                args.output_format,
            )
        )
        saved_paths.extend(
            save_split_files(
                standard_output,
                scenario_dir,
                "StandardScaled",
                args.output_format,
            )
        )
        saved_paths.extend(
            save_split_files(
                minmax_output,
                scenario_dir,
                "MinMaxScaled",
                args.output_format,
            )
        )

        scenario_report = make_scenario_report(
            scenario_name=scenario_name,
            scenario_frame=scenario_frame,
            original_feature_count=len(feature_columns),
            fitted=fitted,
            conversion_counters=conversion_counters,
            transform_stats=transform_stats,
            saved_paths=saved_paths,
        )

        scenario_report.to_excel(
            scenario_dir / "Preprocessing_Report.xlsx",
            index=False,
        )

        scenario_report.insert(0, "Scenario_Name", scenario_name)
        all_summary_frames.append(scenario_report)

        print("Retained features:", len(fitted["retained_features"]))
        print("Removed constants:", len(fitted["constant_features"]))
        print("Scenario folder  :", scenario_dir)

    combined_summary = pd.concat(all_summary_frames, ignore_index=True)
    combined_summary.to_excel(
        output_dir / "All_Scenarios_Preprocessing_Summary.xlsx",
        index=False,
    )

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "original_feature_count": int(len(feature_columns)),
        "iqr_factor": float(args.iqr_factor),
        "output_format": args.output_format,
        "scenarios": {key: list(value) for key, value in SCENARIOS.items()},
        "imputation": "Median fitted on Train only",
        "outlier_handling": (
            "IQR clipping fitted on Train only; zero-IQR features are not clipped"
        ),
        "standard_scaler": "Fitted on cleaned Train only",
        "minmax_scaler": "Fitted on cleaned Train only",
        "negative_values": (
            "Negative non-Cronbach values converted to NaN before Train-median imputation"
        ),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    (output_dir / "Preprocessing_Metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start_time

    print("=" * 70)
    print("Preprocessing completed successfully.")
    print("Output folder :", output_dir)
    print(
        "Summary report:",
        output_dir / "All_Scenarios_Preprocessing_Summary.xlsx",
    )
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
