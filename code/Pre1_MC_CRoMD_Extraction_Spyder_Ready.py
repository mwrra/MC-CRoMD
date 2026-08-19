# -*- coding: utf-8 -*-
"""
Revised MC-CRoMD feature-extraction script.

Purpose:
- Read histopathology images from class folders.
- Divide each image into non-overlapping patches.
- Compute one Cronbach's Alpha plus eight dispersion measures
  for each OpenCV color channel (B, G, R) per patch.
- Save one row per image with reproducibility metadata.

Important:
- OpenCV loads images in BGR order. The exported feature names therefore
  use B, G, and R explicitly.
- No negative or undefined values are converted to absolute values.
- Undefined values remain NaN and are reported in a quality-control log.
- SourceGroupID must later be replaced or verified using a reliable grouping
  strategy before grouped train/validation/test splitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pingouin as pg
import scipy


DISPERSION_NAMES = ("range", "variance", "std", "mad", "Q1", "Q3", "IQR", "cv")
CHANNEL_NAMES = ("B", "G", "R")

# ============================================================
# إعدادات التشغيل المباشر من Spyder
# غيّر هذه القيم فقط عند الانتقال إلى فئة أو حجم رقعة آخر.
# ============================================================
DEFAULT_INPUT_ROOT = Path(
    r"G:\My Research About Lung Canser\OldResult\lung_colon_image_set\lung_image_sets\lung_scc"
)
DEFAULT_OUTPUT = Path(
    r"G:\My Research About Lung Canser\INASS\NewResults\32\32BZ_lung_scc.xlsx"
)
DEFAULT_LABEL = 3
DEFAULT_CLASS_NAME = "lung_scc"
DEFAULT_PATCH_SIZE = 32


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-32 checksum for reproducibility."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_source_group_id(image_path: Path) -> str:
    """
    Temporary source-group identifier.

    LC25000 does not reliably expose original-image parent identifiers in all
    distributed filenames. Until a verified deduplication/grouping stage is
    implemented, each file stem is retained as its own provisional group.
    """
    return image_path.stem


def safe_cronbach_alpha(channels: np.ndarray) -> float:
    """Compute Cronbach's Alpha safely; return NaN when undefined."""
    try:
        frame = pd.DataFrame(channels, columns=["B", "G", "R"])
        alpha, _ = pg.cronbach_alpha(data=frame)
        return float(alpha) if np.isfinite(alpha) else np.nan
    except Exception:
        return np.nan


def channel_statistics(values: np.ndarray) -> dict[str, float]:
    """Compute the eight channel-wise measures used in MC-CRoMD."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return {name: np.nan for name in DISPERSION_NAMES}

    mean_value = float(np.mean(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    sample_std = float(np.std(values, ddof=1))

    return {
        "range": float(np.max(values) - np.min(values)),
        "variance": float(np.var(values, ddof=1)),
        "std": sample_std,
        "mad": float(np.mean(np.abs(values - mean_value))),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "cv": sample_std / mean_value if mean_value != 0 else np.nan,
    }


def extract_image_features(
    image: np.ndarray,
    patch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Extract ordered patch-wise MC-CRoMD features and QC records."""
    height, width = image.shape[:2]

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image dimensions {width}x{height} are not exactly divisible by "
            f"patch size {patch_size}."
        )

    features: dict[str, float] = {}
    qc_rows: list[dict[str, Any]] = []
    patch_index = 1

    # Row-major order: top-to-bottom, then left-to-right.
    for row_start in range(0, height, patch_size):
        for col_start in range(0, width, patch_size):
            patch = image[
                row_start : row_start + patch_size,
                col_start : col_start + patch_size,
            ]

            if patch.shape != (patch_size, patch_size, 3):
                raise ValueError(
                    f"Unexpected patch shape {patch.shape} at patch {patch_index}."
                )

            channel_matrix = patch.reshape(-1, 3).astype(np.float64)
            alpha = safe_cronbach_alpha(channel_matrix)
            features[f"Cronbach_{patch_index}"] = alpha

            for channel_position, channel_name in enumerate(CHANNEL_NAMES):
                stats = channel_statistics(channel_matrix[:, channel_position])
                for measure_name, value in stats.items():
                    features[f"{channel_name}{measure_name}_{patch_index}"] = value

            if not np.isfinite(alpha):
                qc_rows.append(
                    {
                        "PatchIndex": patch_index,
                        "RowStart": row_start,
                        "ColStart": col_start,
                        "Issue": "Undefined Cronbach alpha",
                    }
                )

            patch_index += 1

    expected_patches = (height // patch_size) * (width // patch_size)
    expected_features = expected_patches * 25
    if len(features) != expected_features:
        raise RuntimeError(
            f"Expected {expected_features} features, generated {len(features)}."
        )

    return features, qc_rows


def collect_images(input_root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Collect image files recursively and sort deterministically."""
    extension_set = {ext.lower() for ext in extensions}
    return sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extension_set
    )


def save_run_metadata(
    output_path: Path,
    input_root: Path,
    patch_size: int,
    class_label: int,
    image_count: int,
) -> None:
    metadata = {
        "input_root": str(input_root.resolve()),
        "output_file": str(output_path.resolve()),
        "patch_size": patch_size,
        "class_label": class_label,
        "image_count": image_count,
        "python_version": sys.version,
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "pingouin_version": pg.__version__,
        "channel_order": "BGR (OpenCV)",
        "variance_definition": "sample variance, ddof=1",
        "standard_deviation_definition": "sample standard deviation, ddof=1",
        "mad_definition": "mean absolute deviation from arithmetic mean",
        "patch_order": "row-major: top-to-bottom then left-to-right",
        "source_group_status": "provisional; must be verified before grouped splitting",
    }
    metadata_path = output_path.with_name(f"{output_path.stem}_RunMetadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MC-CRoMD image features.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", type=int, default=DEFAULT_LABEL)
    parser.add_argument("--class-name", default=DEFAULT_CLASS_NAME)
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        choices=(32, 64, 128, 256),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"],
    )
    parser.add_argument("--skip-checksum", action="store_true")

    # parse_known_args يسمح بالتشغيل من Spyder حتى عند وجود معاملات داخلية إضافية.
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("Ignored unknown arguments:", unknown_args)

    # ضمان بقاء المسارات من النوع Path سواء جاءت من القيم الافتراضية أو سطر الأوامر.
    args.input_root = Path(args.input_root)
    args.output = Path(args.output)

    print("Input folder :", args.input_root)
    print("Output file  :", args.output)
    print("Class name   :", args.class_name)
    print("Class label  :", args.label)
    print("Patch size   :", args.patch_size)

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_root}")
    if not args.input_root.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {args.input_root}")

    start_time = time.perf_counter()
    image_paths = collect_images(args.input_root, tuple(args.extensions))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_root}")

    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for image_number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            qc_rows.append(
                {
                    "ImgName": image_path.stem,
                    "ImagePath": str(image_path.resolve()),
                    "Issue": "Image could not be read",
                }
            )
            continue

        height, width = image.shape[:2]
        try:
            features, image_qc = extract_image_features(image, args.patch_size)
        except Exception as exc:
            qc_rows.append(
                {
                    "ImgName": image_path.stem,
                    "ImagePath": str(image_path.resolve()),
                    "Issue": str(exc),
                }
            )
            continue

        row: dict[str, Any] = {
            "ImgName": image_path.stem,
            "FileName": image_path.name,
            "ImagePath": str(image_path.resolve()),
            "ClassName": args.class_name,
            "label": args.label,
            "SourceGroupID": infer_source_group_id(image_path),
            "PatchSize": args.patch_size,
            "ImageWidth": width,
            "ImageHeight": height,
            "ImageSHA256": "" if args.skip_checksum else sha256_file(image_path),
        }
        row.update(features)
        rows.append(row)

        for issue in image_qc:
            issue.update(
                {
                    "ImgName": image_path.stem,
                    "ImagePath": str(image_path.resolve()),
                }
            )
            qc_rows.append(issue)

        print(f"[{image_number}/{len(image_paths)}] Processed: {image_path.name}")

    if not rows:
        raise RuntimeError("No image produced a valid feature row.")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_frame = pd.DataFrame(rows)
    if output_path.suffix.lower() == ".csv":
        feature_frame.to_csv(output_path, index=False)
    else:
        feature_frame.to_excel(output_path, index=False)

    qc_path = output_path.with_name(f"{output_path.stem}_QC.xlsx")
    pd.DataFrame(qc_rows).to_excel(qc_path, index=False)

    save_run_metadata(
        output_path=output_path,
        input_root=args.input_root,
        patch_size=args.patch_size,
        class_label=args.label,
        image_count=len(feature_frame),
    )

    elapsed = time.perf_counter() - start_time
    print(f"Saved {len(feature_frame)} rows to: {output_path}")
    print(f"Quality-control log: {qc_path}")
    print(f"Execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
