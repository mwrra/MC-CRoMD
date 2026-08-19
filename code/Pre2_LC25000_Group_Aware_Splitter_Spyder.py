# -*- coding: utf-8 -*-
"""
LC25000 Group-Aware Splitter
============================
يربط ملف الخصائص بملف LC25000-clean ثم ينشئ تقسيمًا آمنًا على مستوى group_id
بحيث لا تظهر أي صورة ومشتقاتها في أكثر من Train / Validation / Test.

مصمم للعمل مباشرة من Spyder.

التقسيم الافتراضي:
    Train      = 72% تقريبًا
    Validation = 8% تقريبًا
    Test       = 20% تقريبًا

السبب:
    80% Training pool + 20% final Test
    ثم 10% من Training pool للـ Validation
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
#                    إعدادات التشغيل المباشر من Spyder
# =============================================================================

# ملف LC25000-clean الذي حملته من المستودع
GROUPS_CSV = Path(
    r"G:\My Research About Lung Canser\INASS\1-SourceGroupID\lc25000_image_groups.csv")

# ملف الخصائص الناتج من برنامج استخراج الخصائص
FEATURE_FILE = Path(
    r"G:\My Research About Lung Canser\INASS\NewResults\128\128BZ_5Clas.xlsx")

# مجلد حفظ النتائج
OUTPUT_DIR = Path(
    r"G:\My Research About Lung Canser\INASS\1-SourceGroupID\GroupedSplitResults\128_5Clas")

# نسب التقسيم
FINAL_TEST_RATIO = 0.20

# هذه النسبة محسوبة من الجزء المتبقي بعد عزل الاختبار.
# 0.10 من الـ80% المتبقية = 8% من البيانات الكلية.
VALIDATION_RATIO_FROM_REMAINING = 0.10

# تثبيت النتائج وإمكانية إعادة التجربة
RANDOM_SEED = 42

# عند True يتوقف البرنامج إذا وُجدت صورة واحدة غير مطابقة.
STRICT_MATCHING = True


# =============================================================================
#                              دوال مساعدة
# =============================================================================

def normalize_filename(value: object) -> str:
    """إرجاع اسم الملف فقط، بحروف صغيرة، ومن دون مسافات زائدة."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].lower()


def normalize_stem(value: object) -> str:
    """إرجاع اسم الملف من دون الامتداد للمطابقة الاحتياطية."""
    filename = normalize_filename(value)
    return Path(filename).stem.lower()


def read_table(path: Path) -> pd.DataFrame:
    """قراءة CSV أو Excel حسب امتداد الملف."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        # utf-8-sig يدعم ملفات CSV المحفوظة من Excel أيضًا
        return pd.read_csv(path, encoding="utf-8-sig")

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(
        f"Unsupported file type: {suffix}\n"
        "استخدم CSV أو XLSX أو XLSM أو XLS."
    )


def validate_group_file(groups: pd.DataFrame) -> None:
    """التحقق من بنية ملف LC25000-clean."""
    required = {
        "stem",
        "filename",
        "label",
        "tissue",
        "local_cluster_label",
        "group_id",
    }
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(
            "ملف المجموعات لا يحتوي الأعمدة المطلوبة:\n"
            + "\n".join(missing)
        )

    if groups["filename"].isna().any():
        raise ValueError("يوجد اسم ملف مفقود في ملف المجموعات.")

    if groups["group_id"].isna().any():
        raise ValueError("توجد قيمة group_id مفقودة.")

    duplicated = groups["filename"].astype(str).str.lower().duplicated()
    if duplicated.any():
        examples = groups.loc[duplicated, "filename"].head(10).tolist()
        raise ValueError(
            "توجد أسماء صور مكررة في ملف المجموعات، أمثلة:\n"
            + "\n".join(map(str, examples))
        )


def identify_feature_name_column(features: pd.DataFrame) -> str:
    """
    اختيار عمود اسم الصورة من ملف الخصائص.
    الأولوية: FileName ثم ImgName ثم ImagePath.
    """
    for column in ("FileName", "filename", "ImgName", "ImagePath"):
        if column in features.columns:
            return column

    raise ValueError(
        "لم أجد عمود اسم الصورة في ملف الخصائص.\n"
        "يجب أن يحتوي الملف أحد الأعمدة:\n"
        "FileName أو filename أو ImgName أو ImagePath"
    )


def attach_lc25000_groups(
    features: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ربط الخصائص بالمجموعات.

    تتم المطابقة أولًا بالاسم الكامل مع الامتداد، ثم بالـ stem عند الحاجة.
    """
    name_column = identify_feature_name_column(features)

    feature_work = features.copy()
    group_work = groups.copy()

    feature_work["_original_order"] = np.arange(len(feature_work))
    feature_work["_match_filename"] = feature_work[name_column].map(
        normalize_filename
    )
    feature_work["_match_stem"] = feature_work[name_column].map(normalize_stem)

    group_work["_match_filename"] = group_work["filename"].map(
        normalize_filename
    )
    group_work["_match_stem"] = group_work["stem"].map(normalize_stem)

    group_columns = [
        "_match_filename",
        "_match_stem",
        "filename",
        "label",
        "tissue",
        "local_cluster_label",
        "group_id",
    ]

    # المطابقة الأساسية بالاسم الكامل
    merged = feature_work.merge(
        group_work[group_columns],
        on="_match_filename",
        how="left",
        suffixes=("", "_LC25000"),
        validate="many_to_one",
    )

    # الصور غير المطابقة: محاولة ثانية بالاسم من دون الامتداد
    missing_mask = merged["group_id"].isna()

    if missing_mask.any():
        stem_lookup = (
            group_work[
                [
                    "_match_stem",
                    "filename",
                    "label",
                    "tissue",
                    "local_cluster_label",
                    "group_id",
                ]
            ]
            .drop_duplicates(subset="_match_stem", keep=False)
            .set_index("_match_stem")
        )

        for idx in merged.index[missing_mask]:
            stem = merged.at[idx, "_match_stem"]
            if stem in stem_lookup.index:
                row = stem_lookup.loc[stem]
                merged.at[idx, "filename"] = row["filename"]
                merged.at[idx, "label_LC25000"] = row["label"]
                merged.at[idx, "tissue"] = row["tissue"]
                merged.at[idx, "local_cluster_label"] = row[
                    "local_cluster_label"
                ]
                merged.at[idx, "group_id"] = row["group_id"]

    # إعادة تسمية أعمدة المصدر حتى لا تختلط مع label الرقمي في ملف الخصائص
    rename_map = {
        "filename": "LC25000_FileName",
        "label_LC25000": "LC25000_ClassName",
        "tissue": "LC25000_Tissue",
        "local_cluster_label": "LC25000_LocalCluster",
        "group_id": "LC25000_GroupID",
    }

    # عندما لا يوجد عمود label أصلي قد لا يضيف pandas اللاحقة
    if "label" in merged.columns and "label_LC25000" not in merged.columns:
        # هذه الحالة نادرة؛ نحافظ على label الأصلي ونأخذ تسمية المصدر من جديد
        class_map = group_work.set_index("_match_filename")["label"]
        merged["LC25000_ClassName"] = merged["_match_filename"].map(class_map)
        rename_map.pop("label_LC25000", None)

    merged = merged.rename(columns=rename_map)

    unmatched = merged[merged["LC25000_GroupID"].isna()].copy()

    merged = merged.sort_values("_original_order").drop(
        columns=["_original_order", "_match_filename", "_match_stem"],
        errors="ignore",
    )

    unmatched = unmatched.drop(
        columns=["_original_order", "_match_filename", "_match_stem"],
        errors="ignore",
    )

    return merged, unmatched


def split_one_class(
    class_df: pd.DataFrame,
    random_seed: int,
) -> pd.Series:
    """
    تقسيم فئة واحدة على مستوى المجموعات:
    20% اختبار، ثم 10% من المتبقي تحقق.
    """
    if class_df["LC25000_GroupID"].nunique() < 3:
        raise ValueError(
            f"الفئة {class_df['LC25000_ClassName'].iloc[0]} تحتوي أقل من "
            "ثلاث مجموعات، ولا يمكن تقسيمها إلى ثلاثة أقسام."
        )

    split_labels = pd.Series(index=class_df.index, dtype="object")

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=FINAL_TEST_RATIO,
        random_state=random_seed,
    )

    remaining_pos, test_pos = next(
        first_split.split(
            class_df,
            groups=class_df["LC25000_GroupID"],
        )
    )

    remaining_df = class_df.iloc[remaining_pos]
    test_index = class_df.iloc[test_pos].index
    split_labels.loc[test_index] = "Test"

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_RATIO_FROM_REMAINING,
        random_state=random_seed + 1,
    )

    train_pos, validation_pos = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["LC25000_GroupID"],
        )
    )

    train_index = remaining_df.iloc[train_pos].index
    validation_index = remaining_df.iloc[validation_pos].index

    split_labels.loc[train_index] = "Train"
    split_labels.loc[validation_index] = "Validation"

    return split_labels


def create_group_aware_split(data: pd.DataFrame) -> pd.DataFrame:
    """إجراء التقسيم داخل كل فئة للمحافظة على توازن الفئات."""
    if data["LC25000_GroupID"].isna().any():
        raise ValueError(
            "لا يمكن إنشاء التقسيم لوجود صور بلا LC25000_GroupID."
        )

    result = data.copy()
    result["Split"] = ""

    classes = sorted(result["LC25000_ClassName"].dropna().unique())

    for class_number, class_name in enumerate(classes):
        class_mask = result["LC25000_ClassName"] == class_name
        class_df = result.loc[class_mask]

        labels = split_one_class(
            class_df=class_df,
            random_seed=RANDOM_SEED + class_number * 100,
        )
        result.loc[labels.index, "Split"] = labels

    if (result["Split"] == "").any():
        raise RuntimeError("بعض الصفوف لم تحصل على قيمة Split.")

    return result


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """ملخص عدد الصور والمجموعات والنسب لكل فئة وتقسيم."""
    rows = []

    for class_name in sorted(data["LC25000_ClassName"].unique()):
        class_df = data[data["LC25000_ClassName"] == class_name]
        class_total = len(class_df)

        for split_name in ("Train", "Validation", "Test"):
            part = class_df[class_df["Split"] == split_name]
            rows.append(
                {
                    "ClassName": class_name,
                    "Split": split_name,
                    "Images": len(part),
                    "ImagePercentWithinClass": (
                        100.0 * len(part) / class_total if class_total else 0.0
                    ),
                    "Groups": part["LC25000_GroupID"].nunique(),
                }
            )

    # الإجمالي
    total = len(data)
    for split_name in ("Train", "Validation", "Test"):
        part = data[data["Split"] == split_name]
        rows.append(
            {
                "ClassName": "ALL",
                "Split": split_name,
                "Images": len(part),
                "ImagePercentWithinClass": (
                    100.0 * len(part) / total if total else 0.0
                ),
                "Groups": part["LC25000_GroupID"].nunique(),
            }
        )

    return pd.DataFrame(rows)


def audit_group_overlap(data: pd.DataFrame) -> pd.DataFrame:
    """
    يجب أن يظهر كل GroupID في Split واحد فقط.
    يعيد الصفوف المخالفة فقط.
    """
    overlap = (
        data.groupby("LC25000_GroupID")["Split"]
        .nunique()
        .reset_index(name="NumberOfSplits")
    )
    overlap = overlap[overlap["NumberOfSplits"] > 1].copy()

    if overlap.empty:
        return pd.DataFrame(
            columns=[
                "LC25000_GroupID",
                "NumberOfSplits",
                "Splits",
                "Images",
            ]
        )

    details = []
    for group_id in overlap["LC25000_GroupID"]:
        part = data[data["LC25000_GroupID"] == group_id]
        details.append(
            {
                "LC25000_GroupID": group_id,
                "NumberOfSplits": part["Split"].nunique(),
                "Splits": ", ".join(sorted(part["Split"].unique())),
                "Images": len(part),
            }
        )

    return pd.DataFrame(details)


def write_excel_safely(data: pd.DataFrame, path: Path) -> None:
    """حفظ ملف Excel مع رسالة خطأ واضحة عند كونه مفتوحًا."""
    try:
        data.to_excel(path, index=False)
    except PermissionError as exc:
        raise PermissionError(
            f"تعذر حفظ الملف لأنه مفتوح في Excel:\n{path}\n"
            "أغلق الملف ثم أعد التشغيل."
        ) from exc


def main() -> None:
    start = time.perf_counter()

    print("=" * 72)
    print("LC25000 Group-Aware Splitter")
    print("=" * 72)
    print(f"Groups CSV : {GROUPS_CSV}")
    print(f"Features   : {FEATURE_FILE}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Seed       : {RANDOM_SEED}")
    print()

    if not GROUPS_CSV.exists():
        raise FileNotFoundError(
            f"لم أجد ملف المجموعات:\n{GROUPS_CSV}"
        )

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"لم أجد ملف الخصائص:\n{FEATURE_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = read_table(GROUPS_CSV)
    features = read_table(FEATURE_FILE)

    validate_group_file(groups)

    print(f"Rows in groups file  : {len(groups):,}")
    print(f"Unique source groups : {groups['group_id'].nunique():,}")
    print(f"Rows in feature file : {len(features):,}")

    attached, unmatched = attach_lc25000_groups(features, groups)

    matched_count = attached["LC25000_GroupID"].notna().sum()
    print(f"Matched images       : {matched_count:,}")
    print(f"Unmatched images     : {len(unmatched):,}")

    unmatched_path = OUTPUT_DIR / "Unmatched_Images.xlsx"
    write_excel_safely(unmatched, unmatched_path)

    if STRICT_MATCHING and not unmatched.empty:
        raise RuntimeError(
            f"توجد {len(unmatched):,} صورة غير مطابقة.\n"
            f"راجع الملف:\n{unmatched_path}"
        )

    matched = attached[attached["LC25000_GroupID"].notna()].copy()

    # استبدال SourceGroupID التمهيدي بالمعرف المنشور من LC25000-clean
    if "SourceGroupID" in matched.columns:
        matched = matched.rename(
            columns={"SourceGroupID": "Previous_SourceGroupID"}
        )

    insert_position = min(6, len(matched.columns))
    source_group = matched.pop("LC25000_GroupID")
    matched.insert(insert_position, "SourceGroupID", source_group)

    # الاحتفاظ بنسخة صريحة من المعرف نفسه لسهولة التدقيق
    matched["LC25000_GroupID"] = matched["SourceGroupID"]

    split_data = create_group_aware_split(matched)

    summary = build_summary(split_data)
    overlap = audit_group_overlap(split_data)

    if not overlap.empty:
        raise RuntimeError(
            "حدث تداخل في المجموعات بين الأقسام. لم تُحفظ النتائج النهائية."
        )

    # تدقيق إضافي مباشر
    train_groups = set(
        split_data.loc[split_data["Split"] == "Train", "SourceGroupID"]
    )
    val_groups = set(
        split_data.loc[split_data["Split"] == "Validation", "SourceGroupID"]
    )
    test_groups = set(
        split_data.loc[split_data["Split"] == "Test", "SourceGroupID"]
    )

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # حفظ الملفات
    all_path = OUTPUT_DIR / "Features_With_Groups_And_Split.xlsx"
    train_path = OUTPUT_DIR / "Train.xlsx"
    validation_path = OUTPUT_DIR / "Validation.xlsx"
    test_path = OUTPUT_DIR / "Test.xlsx"
    audit_path = OUTPUT_DIR / "Split_Audit.xlsx"

    write_excel_safely(split_data, all_path)
    write_excel_safely(
        split_data[split_data["Split"] == "Train"].copy(),
        train_path,
    )
    write_excel_safely(
        split_data[split_data["Split"] == "Validation"].copy(),
        validation_path,
    )
    write_excel_safely(
        split_data[split_data["Split"] == "Test"].copy(),
        test_path,
    )

    with pd.ExcelWriter(audit_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        overlap.to_excel(writer, sheet_name="Group_Overlap", index=False)
        unmatched.to_excel(writer, sheet_name="Unmatched", index=False)

        settings = pd.DataFrame(
            {
                "Setting": [
                    "FeatureFile",
                    "GroupsCSV",
                    "RandomSeed",
                    "FinalTestRatio",
                    "ValidationRatioFromRemaining",
                    "TrainGroups",
                    "ValidationGroups",
                    "TestGroups",
                    "GroupOverlapCount",
                ],
                "Value": [
                    str(FEATURE_FILE),
                    str(GROUPS_CSV),
                    RANDOM_SEED,
                    FINAL_TEST_RATIO,
                    VALIDATION_RATIO_FROM_REMAINING,
                    len(train_groups),
                    len(val_groups),
                    len(test_groups),
                    len(overlap),
                ],
            }
        )
        settings.to_excel(writer, sheet_name="Settings", index=False)

    run_metadata = {
        "feature_file": str(FEATURE_FILE),
        "groups_csv": str(GROUPS_CSV),
        "output_dir": str(OUTPUT_DIR),
        "random_seed": RANDOM_SEED,
        "final_test_ratio": FINAL_TEST_RATIO,
        "validation_ratio_from_remaining": VALIDATION_RATIO_FROM_REMAINING,
        "matched_images": int(matched_count),
        "unmatched_images": int(len(unmatched)),
        "train_images": int((split_data["Split"] == "Train").sum()),
        "validation_images": int(
            (split_data["Split"] == "Validation").sum()
        ),
        "test_images": int((split_data["Split"] == "Test").sum()),
        "train_groups": len(train_groups),
        "validation_groups": len(val_groups),
        "test_groups": len(test_groups),
        "group_overlap_count": int(len(overlap)),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }

    with open(
        OUTPUT_DIR / "Run_Metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(run_metadata, file, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start

    print()
    print("=" * 72)
    print("تم التنفيذ بنجاح")
    print("=" * 72)
    print(summary.to_string(index=False))
    print()
    print("Group overlap count : 0")
    print(f"Execution time      : {elapsed:.2f} seconds")
    print()
    print("Saved files:")
    print(all_path)
    print(train_path)
    print(validation_path)
    print(test_path)
    print(audit_path)
    print(OUTPUT_DIR / "Run_Metadata.json")


if __name__ == "__main__":
    main()
