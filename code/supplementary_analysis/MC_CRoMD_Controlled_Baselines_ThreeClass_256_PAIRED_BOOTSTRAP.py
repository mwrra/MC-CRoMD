# -*- coding: utf-8 -*-
"""
MC_CRoMD_Controlled_Baselines_ThreeClass_256_REVISED.py
========================================================

Reviewer-requested controlled comparison for MC-CRoMD.

Scientific question
-------------------
Does the MC-CRoMD representation provide an advantage over simpler
conventional handcrafted representations when ONLY the representation changes
and the data split, preprocessing policy, classifier, hyperparameters, and
evaluation procedure are held fixed?

Fixed comparison
----------------
Task       : Three-Class lung classification
Patch size : 256x256
Split      : Existing source-group-aware Train / Validation / locked Test
Classifier : SVM-RBF
C          : 10.0
gamma      : "scale"
class_weight: "balanced"
Random seed: 42

The SVM configuration above is the already validation-selected and frozen
Three-Class configuration used in the final MC-CRoMD ML workflow. It is NOT
retuned for any representation in this controlled experiment.

Representations
---------------
1) MC_CRoMD_All225
   Existing 225-dimensional StandardScaled MC-CRoMD representation.

2) RGB_Histogram_216
   9 non-overlapping 256x256 patches x 3 RGB channels x 8 normalized bins
   = 216 raw handcrafted features.

3) GLCM_216
   9 non-overlapping 256x256 patches x 4 directions x 6 properties
   = 216 raw handcrafted texture features.
   Grayscale quantization: 16 levels; distance=1; symmetric normalized GLCM.
   Properties: contrast, dissimilarity, homogeneity, ASM, energy, correlation.

Leakage safeguards
------------------
- The approved SourceGroupID split is reused exactly.
- Baseline preprocessing is learned from Train only.
- Validation and Test use unchanged Train-derived preprocessing parameters.
- The SVM configuration is fixed before Validation/Test evaluation.
- No feature selection, model selection, hyperparameter tuning, threshold
  tuning, or Train+Validation refit is performed in this program.
- Test is loaded only after baseline extraction/preprocessing objects and the
  fixed comparison configuration have been established.
- 95% bootstrap CIs are SourceGroupID-aware when SourceGroupID is available,
  matching the final MC-CRoMD ML evaluation policy.
- All aggregate and class-wise metrics come from the same predictions.

Preprocessing policy
--------------------
MC-CRoMD:
- Uses the already-approved Pre4 StandardScaled files unchanged.

RGB Histogram / GLCM:
- non-finite -> NaN
- Train-only median imputation
- Train-only removal of all-missing and constant features
- Train-only 1.5*IQR clipping; zero-IQR features are not clipped
- Train-only StandardScaler
- unchanged transform applied to Validation/Test

Note:
Negative GLCM correlation values are statistically valid and are therefore
retained. This is an intentional descriptor-specific exception to the
nonnegative rule used for MC-CRoMD dispersion measures.

Outputs
-------
G:/My Research About Lung Canser/INASS/11-ControlledBaselines/Three_Class_256

Main files:
- Controlled_Baselines_Validation_Fixed_Config.xlsx
- Controlled_Baselines_Final_Locked_Test.xlsx
- Controlled_Baselines_Manuscript_Table.xlsx
- Controlled_Baselines_Metadata.json
- one folder per representation with predictions, class-wise metrics,
  confusion matrix, bootstrap distribution, and 95% CIs.

Run directly from Spyder.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC


# =============================================================================
# FIXED SCIENTIFIC SETTINGS
# =============================================================================

BASE_DIR = Path(r"G:\My Research About Lung Canser\INASS")
LC25000_ROOT = Path(
    r"G:\My Research About Lung Canser\OldResult\lung_colon_image_set"
)

PATCH_SIZE = 256
SCENARIO = "Three_Class"
SOURCE_VARIANT = "StandardScaled"
TARGET_COLUMN = "label"

OUTPUT_DIR = (
    BASE_DIR
    / "11-ControlledBaselines"
    / "Three_Class_256"
)

# Frozen Three-Class SVM-RBF configuration from the approved final ML workflow.
RANDOM_STATE = 42
SVM_C = 10.0
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"
SVM_CACHE_SIZE_MB = 2000

# Baseline feature settings.
RGB_HIST_BINS = 8

GLCM_LEVELS = 16
GLCM_DISTANCE = 1
GLCM_DIRECTIONS = (
    (0, 1, "0deg"),
    (1, 1, "45deg"),
    (1, 0, "90deg"),
    (1, -1, "135deg"),
)
GLCM_PROPERTIES = (
    "contrast",
    "dissimilarity",
    "homogeneity",
    "asm",
    "energy",
    "correlation",
)

IQR_FACTOR = 1.5
EPS = 1e-12

# Match Final_ML_Locked_Test.py defaults.
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260819
CI_LEVEL = 0.95

USE_BASELINE_CACHE = True

SAMPLE_ID_CANDIDATES = ("ImageSHA256", "FileName", "ImgName")
GROUP_ID_CANDIDATES = ("SourceGroupID", "LC25000_GroupID", "group_id")

METADATA_COLUMNS = {
    "ImgName", "FileName", "ImagePath", "ClassName", "label", "SourceGroupID",
    "PatchSize", "ImageWidth", "ImageHeight", "ImageSHA256", "Split", "split",
    "stem", "filename", "tissue", "group_id", "local_cluster_label",
    "LC25000_ClassName", "LC25000_FileName", "LC25000_Tissue",
    "Previous_SourceGroupID", "Previous_SourceGroupID_From_Raw",
    "Previous_SourceGroupID_From_Pre1", "_match_stem_LC25000",
    "LC25000_GroupID", "LC25000_LocalCluster",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =============================================================================
# UTILITIES
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
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def choose_first_existing(
    columns: pd.Index,
    candidates: tuple[str, ...],
) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def normalize_filename(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].lower()


# =============================================================================
# APPROVED MC-CROMD SPLITS
# =============================================================================

def split_folder() -> Path:
    return (
        BASE_DIR
        / "3-DataPreprocessing"
        / str(PATCH_SIZE)
        / SCENARIO
        / SOURCE_VARIANT
    )


def load_split(split_name: str) -> tuple[pd.DataFrame, Path]:
    if split_name not in {"Train", "Validation", "Test"}:
        raise ValueError(split_name)

    path = find_existing_file(split_folder(), split_name)
    frame = read_table(path)

    required = {TARGET_COLUMN, "SourceGroupID"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            f"{split_name}: missing required columns: {sorted(missing)}"
        )

    if "Split" in frame.columns:
        actual = set(frame["Split"].astype(str).str.strip().unique())
        if actual != {split_name}:
            raise RuntimeError(
                f"{split_name}: expected Split={split_name!r}, found {sorted(actual)}"
            )

    return frame, path


def detect_mc_cromd_features(frame: pd.DataFrame) -> list[str]:
    features = [
        str(c)
        for c in frame.columns
        if str(c) not in METADATA_COLUMNS
        and str(c) != TARGET_COLUMN
        and not str(c).startswith("Unnamed:")
    ]
    if len(features) != 225:
        raise RuntimeError(
            f"Expected 225 MC-CRoMD features for 256x256; found {len(features)}."
        )
    if len(features) != len(set(features)):
        raise RuntimeError("Duplicate MC-CRoMD feature names detected.")
    return features


def audit_group_overlap(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> dict[str, int]:
    tr = set(train["SourceGroupID"].astype(str))
    va = set(validation["SourceGroupID"].astype(str))

    audit = {"Train_Validation_Overlap": len(tr & va)}

    if test is not None:
        te = set(test["SourceGroupID"].astype(str))
        audit["Train_Test_Overlap"] = len(tr & te)
        audit["Validation_Test_Overlap"] = len(va & te)

    if any(audit.values()):
        raise RuntimeError(f"SourceGroupID overlap detected: {audit}")

    return audit


# =============================================================================
# IMAGE RESOLUTION
# =============================================================================

_IMAGE_INDEX: dict[str, Path] | None = None


def build_image_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"LC25000_ROOT not found:\n{root}")

    index: dict[str, Path] = {}
    ambiguous: set[str] = set()

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = path.name.lower()
            if key in index and index[key] != path:
                ambiguous.add(key)
            else:
                index[key] = path

    for key in ambiguous:
        index.pop(key, None)

    if not index:
        raise RuntimeError(f"No images found under {root}")

    print(
        f"Image index: {len(index):,} unique filenames; "
        f"{len(ambiguous):,} ambiguous names excluded."
    )
    return index


def choose_image_id_column(frame: pd.DataFrame) -> str:
    for column in ("FileName", "ImgName", "LC25000_FileName"):
        if column in frame.columns:
            return column
    raise KeyError("No filename column found for image resolution.")


def resolve_image_paths(frame: pd.DataFrame, split_name: str) -> list[Path]:
    global _IMAGE_INDEX

    # Prefer the exact stored path if all rows are valid.
    if "ImagePath" in frame.columns:
        direct: list[Path] = []
        valid = True
        for value in frame["ImagePath"]:
            if pd.isna(value):
                valid = False
                break
            path = Path(str(value))
            if not path.is_file():
                valid = False
                break
            direct.append(path)
        if valid and len(direct) == len(frame):
            print(f"{split_name}: using stored ImagePath values.")
            return direct

    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = build_image_index(LC25000_ROOT)

    id_col = choose_image_id_column(frame)
    paths: list[Path] = []
    missing: list[str] = []

    for value in frame[id_col]:
        key = normalize_filename(value)
        path = _IMAGE_INDEX.get(key)
        if path is None:
            missing.append(str(value))
        else:
            paths.append(path)

    if missing:
        raise FileNotFoundError(
            f"{split_name}: {len(missing)} images could not be resolved. "
            f"Examples: {missing[:10]}"
        )

    if len(paths) != len(frame):
        raise RuntimeError(f"{split_name}: resolved image count mismatch.")

    print(f"{split_name}: resolved {len(paths):,} image paths.")
    return paths


# =============================================================================
# RGB HISTOGRAM (216 RAW FEATURES)
# =============================================================================

def rgb_hist_feature_names() -> list[str]:
    names: list[str] = []
    for patch in range(1, 10):
        for channel in ("R", "G", "B"):
            for bin_number in range(1, RGB_HIST_BINS + 1):
                names.append(
                    f"RGBHist_P{patch:02d}_{channel}_Bin{bin_number:02d}"
                )
    if len(names) != 216:
        raise RuntimeError("RGB histogram feature definition is not 216.")
    return names


def extract_rgb_histogram(rgb: np.ndarray) -> np.ndarray:
    values: list[float] = []

    for r0 in range(0, 768, PATCH_SIZE):
        for c0 in range(0, 768, PATCH_SIZE):
            patch = rgb[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            for channel_index in range(3):
                hist, _ = np.histogram(
                    patch[:, :, channel_index],
                    bins=RGB_HIST_BINS,
                    range=(0, 256),
                )
                hist = hist.astype(np.float64)
                total = hist.sum()
                if total <= 0:
                    raise RuntimeError("Unexpected empty RGB histogram.")
                values.extend((hist / total).tolist())

    result = np.asarray(values, dtype=np.float64)
    if result.size != 216:
        raise RuntimeError(f"RGB histogram produced {result.size} features.")
    return result


# =============================================================================
# GLCM (216 RAW FEATURES)
# =============================================================================

def glcm_feature_names() -> list[str]:
    names: list[str] = []
    for patch in range(1, 10):
        for _, _, direction_name in GLCM_DIRECTIONS:
            for prop in GLCM_PROPERTIES:
                names.append(
                    f"GLCM_P{patch:02d}_{direction_name}_{prop}"
                )
    if len(names) != 216:
        raise RuntimeError("GLCM feature definition is not 216.")
    return names


def quantize_gray(gray: np.ndarray) -> np.ndarray:
    q = (gray.astype(np.uint16) * GLCM_LEVELS) // 256
    return np.clip(q, 0, GLCM_LEVELS - 1).astype(np.int16)


def build_symmetric_glcm(
    q: np.ndarray,
    dr: int,
    dc: int,
) -> np.ndarray:
    h, w = q.shape

    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)

    src = q[r0:r1, c0:c1].reshape(-1)
    dst = q[r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(-1)

    if src.size == 0:
        raise RuntimeError("No valid pixel pairs for GLCM.")

    matrix = np.zeros((GLCM_LEVELS, GLCM_LEVELS), dtype=np.float64)
    np.add.at(matrix, (src, dst), 1.0)
    np.add.at(matrix, (dst, src), 1.0)

    total = matrix.sum()
    if total <= 0:
        raise RuntimeError("Empty GLCM.")

    return matrix / total


def calculate_glcm_properties(matrix: np.ndarray) -> tuple[float, ...]:
    levels = matrix.shape[0]
    grid = np.arange(levels, dtype=np.float64)
    i = grid[:, None]
    j = grid[None, :]
    diff = i - j

    contrast = float(np.sum(matrix * diff ** 2))
    dissimilarity = float(np.sum(matrix * np.abs(diff)))
    homogeneity = float(np.sum(matrix / (1.0 + diff ** 2)))
    asm = float(np.sum(matrix ** 2))
    energy = float(math.sqrt(max(asm, 0.0)))

    pi = matrix.sum(axis=1)
    pj = matrix.sum(axis=0)
    mu_i = float(np.sum(grid * pi))
    mu_j = float(np.sum(grid * pj))
    sd_i = float(math.sqrt(max(np.sum((grid - mu_i) ** 2 * pi), 0.0)))
    sd_j = float(math.sqrt(max(np.sum((grid - mu_j) ** 2 * pj), 0.0)))

    if sd_i <= EPS or sd_j <= EPS:
        correlation = 1.0
    else:
        correlation = float(
            np.sum(matrix * (i - mu_i) * (j - mu_j)) / (sd_i * sd_j)
        )

    return (
        contrast,
        dissimilarity,
        homogeneity,
        asm,
        energy,
        correlation,
    )


def extract_glcm(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values: list[float] = []

    for r0 in range(0, 768, PATCH_SIZE):
        for c0 in range(0, 768, PATCH_SIZE):
            patch = gray[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]
            q = quantize_gray(patch)

            for dr, dc, _ in GLCM_DIRECTIONS:
                matrix = build_symmetric_glcm(
                    q,
                    dr * GLCM_DISTANCE,
                    dc * GLCM_DISTANCE,
                )
                values.extend(calculate_glcm_properties(matrix))

    result = np.asarray(values, dtype=np.float64)
    if result.size != 216:
        raise RuntimeError(f"GLCM produced {result.size} features.")
    return result


# =============================================================================
# RAW BASELINE EXTRACTION / CACHE
# =============================================================================

def cache_path(split_name: str, representation: str) -> Path:
    return (
        OUTPUT_DIR
        / "Feature_Cache"
        / f"{representation}_{split_name}_RAW.npz"
    )


def save_raw_cache(
    path: Path,
    x: np.ndarray,
    frame: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_col = choose_first_existing(frame.columns, SAMPLE_ID_CANDIDATES)
    if sample_col is None:
        sample_col = choose_image_id_column(frame)

    np.savez_compressed(
        path,
        X=x,
        SampleID=frame[sample_col].astype(str).to_numpy(),
        SampleColumn=np.asarray([sample_col], dtype=object),
    )


def load_raw_cache(
    path: Path,
    frame: pd.DataFrame,
) -> np.ndarray | None:
    if not (USE_BASELINE_CACHE and path.exists()):
        return None

    saved = np.load(path, allow_pickle=True)
    sample_col = str(saved["SampleColumn"][0])
    if sample_col not in frame.columns:
        raise RuntimeError(
            f"Cache expects sample column {sample_col!r}, not present in split."
        )

    expected = frame[sample_col].astype(str).to_numpy()
    actual = saved["SampleID"].astype(str)

    if len(actual) != len(expected) or not np.array_equal(actual, expected):
        raise RuntimeError(f"Cached sample alignment mismatch: {path}")

    print("Using cached raw baseline:", path.name)
    return np.asarray(saved["X"], dtype=np.float64)


def extract_raw_baselines(
    frame: pd.DataFrame,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_cache = cache_path(split_name, "RGB_Histogram_216")
    glcm_cache = cache_path(split_name, "GLCM_216")

    cached_rgb = load_raw_cache(rgb_cache, frame)
    cached_glcm = load_raw_cache(glcm_cache, frame)

    if cached_rgb is not None and cached_glcm is not None:
        return cached_rgb, cached_glcm

    paths = resolve_image_paths(frame, split_name)
    rgb_matrix = np.empty((len(frame), 216), dtype=np.float64)
    glcm_matrix = np.empty((len(frame), 216), dtype=np.float64)

    for index, path in enumerate(paths):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if (h, w) != (768, 768):
            raise ValueError(f"Expected 768x768, got {w}x{h}: {path}")

        rgb_matrix[index] = extract_rgb_histogram(rgb)
        glcm_matrix[index] = extract_glcm(rgb)

        if (index + 1) % 250 == 0 or index + 1 == len(frame):
            print(
                f"{split_name}: baseline extraction "
                f"{index + 1:,}/{len(frame):,}"
            )

    save_raw_cache(rgb_cache, rgb_matrix, frame)
    save_raw_cache(glcm_cache, glcm_matrix, frame)
    return rgb_matrix, glcm_matrix


# =============================================================================
# TRAIN-ONLY BASELINE PREPROCESSING (MATCHES PRE4 POLICY)
# =============================================================================

def fit_baseline_preprocessor(
    train_raw: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    x = np.asarray(train_raw, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan

    # Remove all-missing features based on Train only.
    all_missing = np.all(np.isnan(x), axis=0)
    candidate_idx = np.flatnonzero(~all_missing)
    if candidate_idx.size == 0:
        raise RuntimeError("All baseline features are missing in Train.")

    x_candidate = x[:, candidate_idx]
    candidate_names = [feature_names[i] for i in candidate_idx]

    medians_candidate = np.nanmedian(x_candidate, axis=0)
    if np.isnan(medians_candidate).any():
        raise RuntimeError("Unexpected NaN Train medians after all-missing removal.")

    x_imputed = np.where(
        np.isnan(x_candidate),
        medians_candidate[None, :],
        x_candidate,
    )

    # Remove constant features after Train-median imputation, as in Pre4.
    constant = np.array(
        [
            np.unique(x_imputed[:, j]).size <= 1
            for j in range(x_imputed.shape[1])
        ],
        dtype=bool,
    )
    retained_local_idx = np.flatnonzero(~constant)
    if retained_local_idx.size == 0:
        raise RuntimeError("All baseline features are constant in Train.")

    retained_original_idx = candidate_idx[retained_local_idx]
    retained_names = [feature_names[i] for i in retained_original_idx]

    x_retained = x[:, retained_original_idx]
    medians = np.nanmedian(x_retained, axis=0)
    x_imputed = np.where(np.isnan(x_retained), medians[None, :], x_retained)

    q1 = np.quantile(x_imputed, 0.25, axis=0)
    q3 = np.quantile(x_imputed, 0.75, axis=0)
    iqr = q3 - q1
    lower = q1 - IQR_FACTOR * iqr
    upper = q3 + IQR_FACTOR * iqr

    zero_iqr = iqr <= 0
    lower[zero_iqr] = -np.inf
    upper[zero_iqr] = np.inf

    x_clipped = np.clip(x_imputed, lower[None, :], upper[None, :])

    scaler = StandardScaler()
    scaler.fit(x_clipped)

    return {
        "original_feature_names": list(feature_names),
        "all_missing_original_indices": np.flatnonzero(all_missing),
        "constant_candidate_indices": np.flatnonzero(constant),
        "retained_original_indices": retained_original_idx,
        "retained_feature_names": retained_names,
        "medians": medians,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "zero_iqr": zero_iqr,
        "scaler": scaler,
    }


def transform_baseline(
    raw: np.ndarray,
    fitted: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    x = np.asarray(raw, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan
    x = x[:, fitted["retained_original_indices"]]

    missing_before = int(np.isnan(x).sum())
    x = np.where(np.isnan(x), fitted["medians"][None, :], x)

    lower_hits = x < fitted["lower"][None, :]
    upper_hits = x > fitted["upper"][None, :]

    x = np.clip(x, fitted["lower"][None, :], fitted["upper"][None, :])
    x = fitted["scaler"].transform(x).astype(np.float32)

    if not np.isfinite(x).all():
        raise RuntimeError("Baseline preprocessing left NaN/inf.")

    return x, {
        "Missing_Before_Imputation": missing_before,
        "Lower_Clipped": int(lower_hits.sum()),
        "Upper_Clipped": int(upper_hits.sum()),
    }


def save_baseline_preprocessor(
    representation: str,
    fitted: dict[str, Any],
) -> None:
    rep_dir = OUTPUT_DIR / representation
    rep_dir.mkdir(parents=True, exist_ok=True)

    retained = fitted["retained_feature_names"]
    pd.DataFrame(
        {
            "Feature": retained,
            "Train_Median": fitted["medians"],
            "Train_Q1": fitted["q1"],
            "Train_Q3": fitted["q3"],
            "Train_IQR": fitted["iqr"],
            "Lower_Bound": fitted["lower"],
            "Upper_Bound": fitted["upper"],
            "Zero_IQR_No_Clipping": fitted["zero_iqr"],
            "StandardScaler_Mean": fitted["scaler"].mean_,
            "StandardScaler_Scale": fitted["scaler"].scale_,
        }
    ).to_excel(
        rep_dir / "Train_Preprocessing_Parameters.xlsx",
        index=False,
    )

    pd.DataFrame(
        {"Retained_Feature": retained}
    ).to_excel(
        rep_dir / "Retained_Features.xlsx",
        index=False,
    )

    joblib.dump(
        fitted,
        rep_dir / "Train_Fitted_Preprocessing.joblib",
        compress=3,
    )


# =============================================================================
# LABELS / MODEL
# =============================================================================

def fit_label_encoder(train: pd.DataFrame) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(train[TARGET_COLUMN].astype(str).to_numpy())
    if len(encoder.classes_) != 3:
        raise RuntimeError(
            f"Expected 3 classes; found {encoder.classes_.tolist()}"
        )
    return encoder


def encode_labels(
    encoder: LabelEncoder,
    frame: pd.DataFrame,
) -> np.ndarray:
    raw = frame[TARGET_COLUMN].astype(str).to_numpy()
    unknown = sorted(set(raw).difference(set(encoder.classes_)))
    if unknown:
        raise RuntimeError(f"Unseen labels: {unknown}")
    return encoder.transform(raw)


def build_display_class_names(
    train: pd.DataFrame,
    encoder: LabelEncoder,
) -> list[str]:
    if "ClassName" not in train.columns:
        return [str(x) for x in encoder.classes_]

    temp = train[[TARGET_COLUMN, "ClassName"]].dropna().copy()
    temp[TARGET_COLUMN] = temp[TARGET_COLUMN].astype(str)
    temp["ClassName"] = temp["ClassName"].astype(str)

    mapping: dict[str, str] = {}
    for label, group in temp.groupby(TARGET_COLUMN):
        names = sorted(group["ClassName"].unique())
        if len(names) == 1:
            mapping[str(label)] = names[0]

    return [mapping.get(str(x), str(x)) for x in encoder.classes_]


def create_fixed_svm() -> SVC:
    # Mirrors Final_ML_Locked_Test.py for the final Three-Class SVM-RBF.
    return SVC(
        kernel="rbf",
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        class_weight=SVM_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        cache_size=SVM_CACHE_SIZE_MB,
    )


# =============================================================================
# METRICS -- MATCH FINAL ML POLICY
# =============================================================================

def get_score_matrix(model: SVC, x: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(x), dtype=float)
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(x), dtype=float)
        except Exception:
            pass
    return None


def calculate_auc(
    y_true: np.ndarray,
    score_matrix: np.ndarray | None,
    classes: np.ndarray,
) -> float:
    if score_matrix is None or score_matrix.ndim != 2:
        return float("nan")
    try:
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
        "ROC_AUC_OVR_Macro": calculate_auc(
            y_true,
            score_matrix,
            classes,
        ),
    }


def classwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=classes,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "Class_Index": classes,
            "Class_Name": class_names,
            "Precision": p,
            "Recall": r,
            "F1": f,
            "Support": s.astype(int),
        }
    )


# =============================================================================
# SOURCE-GROUP-AWARE BOOTSTRAP
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
            idx = np.concatenate(
                [group_to_indices[group] for group in sampled_groups]
            )
        else:
            idx = rng.integers(0, len(y_true), size=len(y_true))

        sampled_scores = (
            None
            if score_matrix is None
            else score_matrix[idx]
        )

        metrics = calculate_metrics(
            y_true[idx],
            y_pred[idx],
            sampled_scores,
            classes,
        )
        rows.append({"Iteration": iteration, **metrics})

    return pd.DataFrame(rows), unit


def confidence_interval_summary(
    bootstrap: pd.DataFrame,
    point_metrics: dict[str, float],
) -> pd.DataFrame:
    alpha = (1.0 - CI_LEVEL) / 2.0
    rows: list[dict[str, Any]] = []

    for metric, point in point_metrics.items():
        values = (
            pd.to_numeric(bootstrap[metric], errors="coerce").dropna()
            if metric in bootstrap.columns
            else pd.Series(dtype=float)
        )
        if values.empty:
            low = high = float("nan")
        else:
            low = float(np.quantile(values, alpha))
            high = float(np.quantile(values, 1.0 - alpha))

        rows.append(
            {
                "Metric": metric,
                "Point_Estimate": float(point),
                "CI_95_Lower": low,
                "CI_95_Upper": high,
            }
        )

    return pd.DataFrame(rows)


def paired_group_bootstrap_difference(
    *,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    scores_a: np.ndarray | None,
    pred_b: np.ndarray,
    scores_b: np.ndarray | None,
    classes: np.ndarray,
    group_values: np.ndarray,
    repeats: int,
    seed: int,
    name_a: str,
    name_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Paired SourceGroupID-aware bootstrap of performance differences.

    The SAME resampled SourceGroupID blocks are used for both representations
    on every bootstrap iteration. This preserves pairing because all methods
    are evaluated on the identical locked-test observations.

    Reported delta is:
        metric(name_a) - metric(name_b)

    A positive CI entirely above 0 supports a performance advantage for A.
    A CI crossing 0 should NOT be described as statistically significant.
    """
    groups = np.asarray(group_values).astype(str)
    unique_groups = np.unique(groups)
    group_to_indices = {
        group: np.flatnonzero(groups == group)
        for group in unique_groups
    }

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for iteration in range(1, repeats + 1):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        idx = np.concatenate(
            [group_to_indices[group] for group in sampled_groups]
        )

        metrics_a = calculate_metrics(
            y_true[idx],
            pred_a[idx],
            None if scores_a is None else scores_a[idx],
            classes,
        )
        metrics_b = calculate_metrics(
            y_true[idx],
            pred_b[idx],
            None if scores_b is None else scores_b[idx],
            classes,
        )

        row = {"Iteration": iteration}
        for metric in (
            "Accuracy",
            "Precision_Macro",
            "Recall_Macro",
            "F1_Macro",
            "Precision_Weighted",
            "Recall_Weighted",
            "F1_Weighted",
            "ROC_AUC_OVR_Macro",
        ):
            row[f"{metric}_{name_a}"] = metrics_a[metric]
            row[f"{metric}_{name_b}"] = metrics_b[metric]
            row[f"Delta_{metric}"] = (
                metrics_a[metric] - metrics_b[metric]
            )
        rows.append(row)

    distribution = pd.DataFrame(rows)

    point_a = calculate_metrics(
        y_true,
        pred_a,
        scores_a,
        classes,
    )
    point_b = calculate_metrics(
        y_true,
        pred_b,
        scores_b,
        classes,
    )

    alpha = (1.0 - CI_LEVEL) / 2.0
    summary_rows: list[dict[str, Any]] = []

    for metric in (
        "Accuracy",
        "Precision_Macro",
        "Recall_Macro",
        "F1_Macro",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "ROC_AUC_OVR_Macro",
    ):
        delta_col = f"Delta_{metric}"
        values = pd.to_numeric(
            distribution[delta_col],
            errors="coerce",
        ).dropna()

        if values.empty:
            ci_low = ci_high = float("nan")
            prob_gt_zero = float("nan")
        else:
            ci_low = float(np.quantile(values, alpha))
            ci_high = float(np.quantile(values, 1.0 - alpha))
            prob_gt_zero = float(np.mean(values > 0.0))

        point_delta = float(point_a[metric] - point_b[metric])

        summary_rows.append(
            {
                "Comparison": f"{name_a} - {name_b}",
                "Metric": metric,
                f"{name_a}_Point": float(point_a[metric]),
                f"{name_b}_Point": float(point_b[metric]),
                "Delta_Point": point_delta,
                "Delta_95CI_Lower": ci_low,
                "Delta_95CI_Upper": ci_high,
                "Bootstrap_Probability_Delta_GT_0": prob_gt_zero,
                "CI_Excludes_Zero": bool(
                    np.isfinite(ci_low)
                    and np.isfinite(ci_high)
                    and (ci_low > 0.0 or ci_high < 0.0)
                ),
                "Direction": (
                    f"{name_a} higher"
                    if point_delta > 0
                    else f"{name_b} higher"
                    if point_delta < 0
                    else "No point-estimate difference"
                ),
            }
        )

    return distribution, pd.DataFrame(summary_rows)


# =============================================================================
# ONE REPRESENTATION / ONE SPLIT
# =============================================================================

def evaluate_model(
    *,
    representation: str,
    model: SVC,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    eval_frame: pd.DataFrame,
    encoder: LabelEncoder,
    class_names: list[str],
    split_name: str,
    save_detailed: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    classes = np.arange(len(encoder.classes_))

    pred = model.predict(x_eval)
    scores = get_score_matrix(model, x_eval)
    metrics = calculate_metrics(y_eval, pred, scores, classes)

    if save_detailed:
        rep_dir = OUTPUT_DIR / representation
        rep_dir.mkdir(parents=True, exist_ok=True)

        sample_col = choose_first_existing(
            eval_frame.columns,
            SAMPLE_ID_CANDIDATES,
        )
        group_col = choose_first_existing(
            eval_frame.columns,
            GROUP_ID_CANDIDATES,
        )

        predictions = pd.DataFrame(
            {
                "Row_Index": np.arange(len(eval_frame)),
                "True_Encoded": y_eval,
                "Predicted_Encoded": pred,
                "True_Label": [class_names[i] for i in y_eval],
                "Predicted_Label": [class_names[i] for i in pred],
            }
        )

        if sample_col is not None:
            predictions.insert(
                1,
                sample_col,
                eval_frame[sample_col].astype(str).to_numpy(),
            )
        if group_col is not None and group_col not in predictions.columns:
            predictions.insert(
                2 if sample_col is not None else 1,
                group_col,
                eval_frame[group_col].astype(str).to_numpy(),
            )

        if scores is not None and scores.ndim == 2:
            for i, class_name in enumerate(class_names):
                predictions[f"Score_{class_name}"] = scores[:, i]

        predictions.to_csv(
            rep_dir / f"{split_name}_Predictions.csv",
            index=False,
        )

        cm = confusion_matrix(y_eval, pred, labels=classes)
        pd.DataFrame(
            cm,
            index=[f"True_{x}" for x in class_names],
            columns=[f"Pred_{x}" for x in class_names],
        ).to_excel(
            rep_dir / f"{split_name}_Confusion_Matrix.xlsx"
        )

        classwise_metrics(
            y_eval,
            pred,
            classes,
            class_names,
        ).to_excel(
            rep_dir / f"{split_name}_Classwise_Metrics.xlsx",
            index=False,
        )

    return metrics, pred, scores


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("MC-CRoMD CONTROLLED BASELINES - REVISED")
    print("Three-Class | 256x256 | fixed SVM-RBF C=10, gamma=scale")
    print("class_weight=balanced | SourceGroupID-aware evaluation")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # PHASE 1: Train + Validation only. Test is not loaded here.
    # -------------------------------------------------------------------------
    train, train_path = load_split("Train")
    validation, validation_path = load_split("Validation")

    overlap_pretest = audit_group_overlap(train, validation)
    print("Train/Validation SourceGroupID overlap: 0")

    mc_train_features = detect_mc_cromd_features(train)
    mc_val_features = detect_mc_cromd_features(validation)
    if mc_train_features != mc_val_features:
        raise RuntimeError("MC-CRoMD Train/Validation feature order differs.")

    x_mc_train = train[mc_train_features].to_numpy(dtype=np.float32)
    x_mc_val = validation[mc_train_features].to_numpy(dtype=np.float32)
    if not np.isfinite(x_mc_train).all() or not np.isfinite(x_mc_val).all():
        raise RuntimeError("MC-CRoMD Train/Validation contains NaN/inf.")

    # Baseline feature extraction on the same Train/Validation images.
    rgb_train_raw, glcm_train_raw = extract_raw_baselines(train, "Train")
    rgb_val_raw, glcm_val_raw = extract_raw_baselines(
        validation,
        "Validation",
    )

    rgb_preprocessor = fit_baseline_preprocessor(
        rgb_train_raw,
        rgb_hist_feature_names(),
    )
    glcm_preprocessor = fit_baseline_preprocessor(
        glcm_train_raw,
        glcm_feature_names(),
    )

    # At this point all Train-derived preprocessing parameters are fixed.
    save_baseline_preprocessor(
        "RGB_Histogram_216",
        rgb_preprocessor,
    )
    save_baseline_preprocessor(
        "GLCM_216",
        glcm_preprocessor,
    )

    x_rgb_train, rgb_train_stats = transform_baseline(
        rgb_train_raw,
        rgb_preprocessor,
    )
    x_rgb_val, rgb_val_stats = transform_baseline(
        rgb_val_raw,
        rgb_preprocessor,
    )
    x_glcm_train, glcm_train_stats = transform_baseline(
        glcm_train_raw,
        glcm_preprocessor,
    )
    x_glcm_val, glcm_val_stats = transform_baseline(
        glcm_val_raw,
        glcm_preprocessor,
    )

    encoder = fit_label_encoder(train)
    y_train = encode_labels(encoder, train)
    y_val = encode_labels(encoder, validation)
    class_names = build_display_class_names(train, encoder)

    train_val_sets = {
        "MC_CRoMD_All225": (x_mc_train, x_mc_val),
        "RGB_Histogram_216": (x_rgb_train, x_rgb_val),
        "GLCM_216": (x_glcm_train, x_glcm_val),
    }

    # Fixed configuration; no validation-based selection occurs here.
    fixed_config = {
        "Model": "SVM_RBF",
        "C": SVM_C,
        "Gamma": SVM_GAMMA,
        "Class_Weight": SVM_CLASS_WEIGHT,
        "Random_State": RANDOM_STATE,
    }
    json_dump(
        OUTPUT_DIR / "Fixed_Comparison_Configuration.json",
        fixed_config,
    )

    validation_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, SVC] = {}

    for representation, (xtr, xva) in train_val_sets.items():
        print(
            f"{representation}: Train={xtr.shape}, Validation={xva.shape}"
        )

        model = create_fixed_svm()
        fit_start = time.perf_counter()
        model.fit(xtr, y_train)
        fit_seconds = time.perf_counter() - fit_start

        metrics, _, _ = evaluate_model(
            representation=representation,
            model=model,
            x_eval=xva,
            y_eval=y_val,
            eval_frame=validation,
            encoder=encoder,
            class_names=class_names,
            split_name="Validation",
            save_detailed=False,
        )

        validation_rows.append(
            {
                "Representation": representation,
                "Raw_Feature_Count": (
                    225 if representation == "MC_CRoMD_All225" else 216
                ),
                "Retained_Feature_Count": int(xtr.shape[1]),
                "Fit_Seconds": fit_seconds,
                **fixed_config,
                **metrics,
            }
        )

        # Save the fixed Train-fitted model BEFORE Test is loaded.
        rep_dir = OUTPUT_DIR / representation
        rep_dir.mkdir(parents=True, exist_ok=True)
        model_path = rep_dir / "Fixed_Train_Model_Before_Test.joblib"
        joblib.dump(
            {
                "model": model,
                "label_encoder": encoder,
                "representation": representation,
                "fixed_config": fixed_config,
                "fit_seconds": fit_seconds,
            },
            model_path,
            compress=3,
        )
        fitted_models[representation] = model

    pd.DataFrame(validation_rows).to_excel(
        OUTPUT_DIR / "Controlled_Baselines_Validation_Fixed_Config.xlsx",
        index=False,
    )

    print()
    print("=" * 100)
    print("PRE-TEST STAGE COMPLETED.")
    print("Models and preprocessing are fixed. Loading locked Test only now.")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # PHASE 2: Locked Test. No selection/tuning/refit after this point.
    # -------------------------------------------------------------------------
    test, test_path = load_split("Test")
    overlap_full = audit_group_overlap(train, validation, test)
    print("Train/Test SourceGroupID overlap: 0")
    print("Validation/Test SourceGroupID overlap: 0")

    mc_test_features = detect_mc_cromd_features(test)
    if mc_test_features != mc_train_features:
        raise RuntimeError("MC-CRoMD Test feature order differs from Train.")

    x_mc_test = test[mc_train_features].to_numpy(dtype=np.float32)
    if not np.isfinite(x_mc_test).all():
        raise RuntimeError("MC-CRoMD Test contains NaN/inf.")

    rgb_test_raw, glcm_test_raw = extract_raw_baselines(test, "Test")
    x_rgb_test, rgb_test_stats = transform_baseline(
        rgb_test_raw,
        rgb_preprocessor,
    )
    x_glcm_test, glcm_test_stats = transform_baseline(
        glcm_test_raw,
        glcm_preprocessor,
    )

    y_test = encode_labels(encoder, test)

    test_sets = {
        "MC_CRoMD_All225": x_mc_test,
        "RGB_Histogram_216": x_rgb_test,
        "GLCM_216": x_glcm_test,
    }

    final_rows: list[dict[str, Any]] = []
    locked_predictions: dict[str, np.ndarray] = {}
    locked_scores: dict[str, np.ndarray | None] = {}

    group_col = choose_first_existing(test.columns, GROUP_ID_CANDIDATES)
    group_values = (
        test[group_col].astype(str).to_numpy()
        if group_col is not None
        else None
    )
    bootstrap_unit_expected = (
        group_col if group_col is not None else "sample"
    )

    classes = np.arange(len(encoder.classes_))

    for representation, x_test in test_sets.items():
        model = fitted_models[representation]

        predict_start = time.perf_counter()
        metrics, pred, scores = evaluate_model(
            representation=representation,
            model=model,
            x_eval=x_test,
            y_eval=y_test,
            eval_frame=test,
            encoder=encoder,
            class_names=class_names,
            split_name="Final_Test",
            save_detailed=True,
        )
        predict_seconds = time.perf_counter() - predict_start

        locked_predictions[representation] = np.asarray(pred).copy()
        locked_scores[representation] = (
            None if scores is None else np.asarray(scores).copy()
        )

        bootstrap, bootstrap_unit = bootstrap_distribution(
            y_true=y_test,
            y_pred=pred,
            score_matrix=scores,
            classes=classes,
            group_values=group_values,
            repeats=BOOTSTRAP_REPEATS,
            seed=BOOTSTRAP_SEED,
        )

        rep_dir = OUTPUT_DIR / representation
        bootstrap.to_csv(
            rep_dir / "Final_Test_Bootstrap_Distribution.csv",
            index=False,
        )

        ci = confidence_interval_summary(bootstrap, metrics)
        ci.to_excel(
            rep_dir / "Final_Test_Metrics_With_95CI.xlsx",
            index=False,
        )

        ci_lookup = ci.set_index("Metric")
        final_rows.append(
            {
                "Representation": representation,
                "Raw_Feature_Count": (
                    225 if representation == "MC_CRoMD_All225" else 216
                ),
                "Retained_Feature_Count": int(
                    train_val_sets[representation][0].shape[1]
                ),
                **fixed_config,
                "Test_Rows": int(len(test)),
                "Predict_Seconds": predict_seconds,
                **metrics,
                "Accuracy_CI_Lower": float(
                    ci_lookup.at["Accuracy", "CI_95_Lower"]
                ),
                "Accuracy_CI_Upper": float(
                    ci_lookup.at["Accuracy", "CI_95_Upper"]
                ),
                "F1_Macro_CI_Lower": float(
                    ci_lookup.at["F1_Macro", "CI_95_Lower"]
                ),
                "F1_Macro_CI_Upper": float(
                    ci_lookup.at["F1_Macro", "CI_95_Upper"]
                ),
                "ROC_AUC_CI_Lower": float(
                    ci_lookup.at["ROC_AUC_OVR_Macro", "CI_95_Lower"]
                ),
                "ROC_AUC_CI_Upper": float(
                    ci_lookup.at["ROC_AUC_OVR_Macro", "CI_95_Upper"]
                ),
                "Bootstrap_Unit": bootstrap_unit,
            }
        )


    # -------------------------------------------------------------------------
    # PAIRED SourceGroupID-aware bootstrap differences
    # -------------------------------------------------------------------------
    if group_values is None:
        raise RuntimeError(
            "Paired bootstrap requires SourceGroupID/group identifier, "
            "but none was found in the locked Test split."
        )

    paired_dir = OUTPUT_DIR / "Paired_Bootstrap_Differences"
    paired_dir.mkdir(parents=True, exist_ok=True)

    comparisons = (
        ("MC_CRoMD_All225", "RGB_Histogram_216"),
        ("MC_CRoMD_All225", "GLCM_216"),
    )

    paired_summaries: list[pd.DataFrame] = []

    for name_a, name_b in comparisons:
        distribution, summary = paired_group_bootstrap_difference(
            y_true=y_test,
            pred_a=locked_predictions[name_a],
            scores_a=locked_scores[name_a],
            pred_b=locked_predictions[name_b],
            scores_b=locked_scores[name_b],
            classes=classes,
            group_values=group_values,
            repeats=BOOTSTRAP_REPEATS,
            seed=BOOTSTRAP_SEED,
            name_a=name_a,
            name_b=name_b,
        )

        safe_name = f"{name_a}_vs_{name_b}"
        distribution.to_csv(
            paired_dir / f"{safe_name}_Paired_Bootstrap_Distribution.csv",
            index=False,
        )
        summary.to_excel(
            paired_dir / f"{safe_name}_Paired_Bootstrap_Summary.xlsx",
            index=False,
        )
        paired_summaries.append(summary)

    paired_summary_all = pd.concat(
        paired_summaries,
        ignore_index=True,
    )
    paired_summary_all.to_excel(
        OUTPUT_DIR / "Controlled_Baselines_Paired_Bootstrap_Summary.xlsx",
        index=False,
    )

    manuscript_paired = paired_summary_all.loc[
        paired_summary_all["Metric"].isin(
            ["Accuracy", "F1_Macro", "ROC_AUC_OVR_Macro"]
        ),
        [
            "Comparison",
            "Metric",
            "Delta_Point",
            "Delta_95CI_Lower",
            "Delta_95CI_Upper",
            "Bootstrap_Probability_Delta_GT_0",
            "CI_Excludes_Zero",
        ],
    ].copy()

    manuscript_paired.to_excel(
        OUTPUT_DIR / "Controlled_Baselines_Paired_Bootstrap_Manuscript.xlsx",
        index=False,
    )

    final = pd.DataFrame(final_rows)
    final.to_excel(
        OUTPUT_DIR / "Controlled_Baselines_Final_Locked_Test.xlsx",
        index=False,
    )

    manuscript_columns = [
        "Representation",
        "Raw_Feature_Count",
        "Retained_Feature_Count",
        "Accuracy",
        "Accuracy_CI_Lower",
        "Accuracy_CI_Upper",
        "Precision_Macro",
        "Recall_Macro",
        "F1_Macro",
        "F1_Macro_CI_Lower",
        "F1_Macro_CI_Upper",
        "Precision_Weighted",
        "Recall_Weighted",
        "F1_Weighted",
        "ROC_AUC_OVR_Macro",
        "ROC_AUC_CI_Lower",
        "ROC_AUC_CI_Upper",
    ]
    final[manuscript_columns].to_excel(
        OUTPUT_DIR / "Controlled_Baselines_Manuscript_Table.xlsx",
        index=False,
    )

    # Feature definitions for reproducibility.
    pd.DataFrame(
        {
            "Rank": np.arange(1, 226),
            "Feature": mc_train_features,
        }
    ).to_excel(
        OUTPUT_DIR / "MC_CRoMD_All225_Features.xlsx",
        index=False,
    )
    pd.DataFrame(
        {
            "Rank": np.arange(1, 217),
            "Feature": rgb_hist_feature_names(),
        }
    ).to_excel(
        OUTPUT_DIR / "RGB_Histogram_216_Raw_Features.xlsx",
        index=False,
    )
    pd.DataFrame(
        {
            "Rank": np.arange(1, 217),
            "Feature": glcm_feature_names(),
        }
    ).to_excel(
        OUTPUT_DIR / "GLCM_216_Raw_Features.xlsx",
        index=False,
    )

    metadata = {
        "generated_at": timestamp(),
        "purpose": (
            "Controlled reviewer-requested comparison isolating the feature "
            "representation while holding split, preprocessing policy, "
            "classifier, hyperparameters, and evaluation procedure fixed."
        ),
        "scenario": SCENARIO,
        "patch_size": PATCH_SIZE,
        "fixed_classifier": fixed_config,
        "reason_for_fixed_classifier": (
            "SVM-RBF C=10, gamma=scale is the already validation-selected "
            "Three-Class final ML configuration in the manuscript/final locked "
            "test workflow. The same configuration is applied unchanged to all "
            "representations; no baseline-specific tuning is performed."
        ),
        "representations": {
            "MC_CRoMD_All225": {
                "raw_features": 225,
                "retained_features": int(x_mc_train.shape[1]),
                "preprocessing": (
                    "Existing approved Pre4 StandardScaled data."
                ),
            },
            "RGB_Histogram_216": {
                "raw_features": 216,
                "retained_features": int(x_rgb_train.shape[1]),
                "definition": (
                    "9 patches x 3 RGB channels x 8 normalized histogram bins."
                ),
            },
            "GLCM_216": {
                "raw_features": 216,
                "retained_features": int(x_glcm_train.shape[1]),
                "definition": (
                    "9 patches x 4 directions x 6 GLCM properties; grayscale "
                    "quantized to 16 levels; distance=1; symmetric normalized GLCM."
                ),
                "properties": list(GLCM_PROPERTIES),
                "valid_negative_rule": (
                    "Negative GLCM correlation is retained as statistically valid."
                ),
            },
        },
        "baseline_preprocessing": (
            "Train-only nonfinite handling, all-missing removal, median "
            "imputation, constant-feature removal, 1.5*IQR clipping with "
            "zero-IQR no-clipping, and StandardScaler. Train parameters are "
            "applied unchanged to Validation and Test."
        ),
        "split_paths": {
            "Train": str(train_path),
            "Validation": str(validation_path),
            "Test": str(test_path),
        },
        "split_counts": {
            "Train": int(len(train)),
            "Validation": int(len(validation)),
            "Test": int(len(test)),
        },
        "group_overlap_audit": overlap_full,
        "evaluation_metrics": [
            "Accuracy",
            "Precision_Macro",
            "Recall_Macro",
            "F1_Macro",
            "Precision_Weighted",
            "Recall_Weighted",
            "F1_Weighted",
            "ROC_AUC_OVR_Macro",
            "class-wise precision/recall/F1/support",
            "confusion matrix",
        ],
        "bootstrap": {
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "unit": bootstrap_unit_expected,
            "policy": (
                "SourceGroupID-aware when available; sample-level fallback only "
                "when no group identifier exists."
            ),
        },
        "test_policy": (
            "No representation-specific model selection or hyperparameter tuning. "
            "Train-fitted preprocessing and models are fixed before Test is loaded. "
            "No Train+Validation refit is performed."
        ),
        "paired_bootstrap": {
            "performed": True,
            "comparisons": [
                "MC_CRoMD_All225 - RGB_Histogram_216",
                "MC_CRoMD_All225 - GLCM_216",
            ],
            "unit": bootstrap_unit_expected,
            "repeats": BOOTSTRAP_REPEATS,
            "seed": BOOTSTRAP_SEED,
            "interpretation": (
                "Delta = MC-CRoMD metric minus baseline metric. "
                "A 95% CI excluding zero supports a paired performance difference."
            ),
        },
        "transform_stats": {
            "RGB_Histogram_216": {
                "Train": rgb_train_stats,
                "Validation": rgb_val_stats,
                "Test": rgb_test_stats,
            },
            "GLCM_216": {
                "Train": glcm_train_stats,
                "Validation": glcm_val_stats,
                "Test": glcm_test_stats,
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "opencv": cv2.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "elapsed_seconds": time.perf_counter() - start,
    }

    json_dump(
        OUTPUT_DIR / "Controlled_Baselines_Metadata.json",
        metadata,
    )

    print()
    print("=" * 100)
    print("CONTROLLED BASELINE COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(final.to_string(index=False))
    print()
    print("Main manuscript-ready table:")
    print(OUTPUT_DIR / "Controlled_Baselines_Manuscript_Table.xlsx")
    print("Paired bootstrap summary:")
    print(OUTPUT_DIR / "Controlled_Baselines_Paired_Bootstrap_Summary.xlsx")
    print()
    print(
        "IMPORTANT: all three representations used the same fixed "
        "SVM-RBF C=10, gamma=scale, class_weight=balanced."
    )
    print(
        "Bootstrap unit:",
        bootstrap_unit_expected,
    )


if __name__ == "__main__":
    main()
