# -*- coding: utf-8 -*-
"""
Generate_Final_Publication_Figures.py

Creates two compact publication figures from ALREADY-SAVED MC-CRoMD outputs.

SCIENTIFIC POLICY
-----------------
- No training, tuning, ranking, model selection, or testing is performed.
- Raw Train/Validation/Test feature matrices are never loaded.
- Only saved DNN training histories and final locked-test confusion matrices are read.
- Therefore this script cannot change any frozen model decision or final result.

Outputs
-------
Figure_7_DNN_Validation_Loss.png / .pdf
Figure_8_Final_Locked_Test_Confusion_Matrices.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

DNN_CONFIGS = {
    "Binary": {
        "patch_size": 64,
        "feature_method": "Mutclass",
        "k": 3600,
        "seed": 42,
        "display": "Binary",
    },
    "Three_Class": {
        "patch_size": 256,
        "feature_method": "Variance",
        "k": 225,
        "seed": 42,
        "display": "Three-Class",
    },
    "Five_Class": {
        "patch_size": 256,
        "feature_method": "Logistic",
        "k": 200,
        "seed": 42,
        "display": "Five-Class",
    },
}

SCENARIO_ORDER = ("Binary", "Three_Class", "Five_Class")
DEFAULT_BASE_DIR = Path(r"G:\My Research About Lung Canser\INASS")
DEFAULT_OUTPUT_FOLDER = "9-PublicationFigures"


def require_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n{path}\n"
            "No figure was fabricated. Check the project path and saved outputs."
        )
    return path


def dnn_history_path(base_dir: Path, scenario: str) -> Path:
    cfg = DNN_CONFIGS[scenario]
    return (
        base_dir
        / "6-DNNValidation"
        / str(cfg["patch_size"])
        / "Runs"
        / scenario
        / cfg["feature_method"]
        / f"K{int(cfg['k']):05d}"
        / f"Seed_{int(cfg['seed'])}"
        / "Training_History.csv"
    )


def final_cm_path(base_dir: Path, family: str, scenario: str) -> Path:
    if family == "DNN":
        root = base_dir / "7-DNNFinalLockedTest"
        filename = "Final_Test_Confusion_Matrix.xlsx"
    elif family == "ML":
        root = base_dir / "8-MLFinalLockedTest"
        filename = "Final_ML_Test_Confusion_Matrix.xlsx"
    else:
        raise ValueError(f"Unknown family: {family}")

    return root / scenario / filename


def read_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Epoch" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Epoch"})
    required = {"Epoch", "loss", "val_loss"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(
            f"{path}\nMissing history columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )
    for col in ("Epoch", "loss", "val_loss"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not np.isfinite(df[["Epoch", "loss", "val_loss"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values detected in {path}")
    return df


def read_confusion_matrix(path: Path) -> tuple[np.ndarray, list[str]]:
    df = pd.read_excel(path, index_col=0)
    matrix = df.to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Confusion matrix is not square: {path}, shape={matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite confusion-matrix values in {path}")
    labels = [x[5:] if x.startswith("True_") else x for x in df.index.astype(str)]
    return matrix.astype(int), labels


def save_figure(fig, output_dir: Path, stem: str) -> None:
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def make_figure_7(base_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for scenario in SCENARIO_ORDER:
        cfg = DNN_CONFIGS[scenario]
        history = read_history(
            require_file(
                dnn_history_path(base_dir, scenario),
                f"DNN training history for {scenario}",
            )
        )
        label = (
            f"{cfg['display']} "
            f"(BZ={cfg['patch_size']}, {cfg['feature_method']}, K={cfg['k']})"
        )
        ax.plot(history["Epoch"], history["val_loss"], linewidth=1.8, label=label)
        best_idx = history["val_loss"].idxmin()
        ax.scatter(
            [float(history.loc[best_idx, "Epoch"])],
            [float(history.loc[best_idx, "val_loss"])],
            s=28,
            zorder=3,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation loss", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "Figure_7_DNN_Validation_Loss")
    plt.close(fig)


def draw_cm_counts(ax, matrix: np.ndarray, labels: list[str], title: str) -> None:
    """Draw confusion matrix using exact sample counts."""
    ax.imshow(matrix, aspect="equal", cmap="Blues")

    n = matrix.shape[0]
    max_value = float(matrix.max()) if matrix.size else 1.0
    threshold = max_value * 0.50

    for i in range(n):
        for j in range(n):
            value = int(matrix[i, j])

            if value >= threshold:
                text_color = "white"
                stroke_color = "black"
            else:
                text_color = "black"
                stroke_color = "white"

            txt = ax.text(
                j, i, f"{value}",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color,
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=1.4, foreground=stroke_color)
            ])

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Predicted class", fontsize=10.5)
    ax.set_ylabel("True class", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, pad=8)


def draw_cm_percent(ax, matrix: np.ndarray, labels: list[str], title: str) -> None:
    """
    Draw row-normalized confusion matrix as percentages.
    Each row sums to 100%; diagonal cells are class-wise recall/sensitivity,
    not overall model accuracy.
    """
    row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
    percentages = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    ax.imshow(percentages, aspect="equal", cmap="Blues", vmin=0, vmax=100)

    n = matrix.shape[0]
    threshold = 50.0

    for i in range(n):
        for j in range(n):
            value = float(percentages[i, j])

            if value >= threshold:
                text_color = "white"
                stroke_color = "black"
            else:
                text_color = "black"
                stroke_color = "white"

            txt = ax.text(
                j, i, f"{value:.1f}%",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color,
            )
            txt.set_path_effects([
                pe.withStroke(linewidth=1.4, foreground=stroke_color)
            ])

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Predicted class", fontsize=10.5)
    ax.set_ylabel("True class", fontsize=10.5)
    ax.set_title(title, fontsize=11.5, pad=8)


def make_figure_8_counts(base_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(11.5, 7.2), constrained_layout=True
    )

    for col, scenario in enumerate(SCENARIO_ORDER):
        display = DNN_CONFIGS[scenario]["display"]

        dnn_path = require_file(
            final_cm_path(base_dir, "DNN", scenario),
            f"final DNN confusion matrix for {scenario}",
        )
        dnn_matrix, dnn_labels = read_confusion_matrix(dnn_path)
        draw_cm_counts(
            axes[0, col], dnn_matrix, dnn_labels, f"DNN – {display}"
        )

        ml_path = require_file(
            final_cm_path(base_dir, "ML", scenario),
            f"final ML confusion matrix for {scenario}",
        )
        ml_matrix, ml_labels = read_confusion_matrix(ml_path)
        draw_cm_counts(
            axes[1, col], ml_matrix, ml_labels, f"ML – {display}"
        )

    fig.suptitle(
        "Final locked-test confusion matrices – exact counts",
        fontsize=12.5,
    )
    save_figure(
        fig,
        output_dir,
        "Figure_8_Final_Locked_Test_Confusion_Matrices_Counts",
    )
    plt.close(fig)


def make_figure_8_percent(base_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2, 3, figsize=(11.5, 7.2), constrained_layout=True
    )

    for col, scenario in enumerate(SCENARIO_ORDER):
        display = DNN_CONFIGS[scenario]["display"]

        dnn_path = require_file(
            final_cm_path(base_dir, "DNN", scenario),
            f"final DNN confusion matrix for {scenario}",
        )
        dnn_matrix, dnn_labels = read_confusion_matrix(dnn_path)
        draw_cm_percent(
            axes[0, col], dnn_matrix, dnn_labels, f"DNN – {display}"
        )

        ml_path = require_file(
            final_cm_path(base_dir, "ML", scenario),
            f"final ML confusion matrix for {scenario}",
        )
        ml_matrix, ml_labels = read_confusion_matrix(ml_path)
        draw_cm_percent(
            axes[1, col], ml_matrix, ml_labels, f"ML – {display}"
        )

    fig.suptitle(
        "Final locked-test confusion matrices – row-normalized percentages",
        fontsize=12.5,
    )
    save_figure(
        fig,
        output_dir,
        "Figure_8_Final_Locked_Test_Confusion_Matrices_Percent",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures from frozen MC-CRoMD outputs."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    args, unknown = parser.parse_known_args()

    if unknown:
        print("Ignored unknown arguments:", unknown)

    base_dir = Path(args.base_dir)
    output_dir = base_dir / DEFAULT_OUTPUT_FOLDER
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("MC-CRoMD PUBLICATION FIGURE GENERATOR")
    print("=" * 92)
    print("Project root :", base_dir)
    print("Output       :", output_dir)
    print("Policy       : READ-ONLY visualization of already-saved results.")
    print("=" * 92)

    print("\nPreflight:")
    for scenario in SCENARIO_ORDER:
        print(" DNN history:", require_file(dnn_history_path(base_dir, scenario), f"DNN history {scenario}"))
        print(" DNN CM     :", require_file(final_cm_path(base_dir, "DNN", scenario), f"DNN final CM {scenario}"))
        print(" ML CM      :", require_file(final_cm_path(base_dir, "ML", scenario), f"ML final CM {scenario}"))

    print("\nGenerating Figure 7...")
    make_figure_7(base_dir, output_dir)

    print("\nGenerating Figure 8 (exact counts)...")
    make_figure_8_counts(base_dir, output_dir)

    print("\nGenerating Figure 8 (row-normalized percentages)...")
    make_figure_8_percent(base_dir, output_dir)

    print("\n" + "=" * 92)
    print("PUBLICATION FIGURES GENERATED SUCCESSFULLY")
    print("=" * 92)
    print("Figure 7 caption:")
    print("Validation-loss trajectories of the frozen DNN configurations selected before final locked-test evaluation.")
    print("\nFigure 8 caption:")
    print("Confusion matrices of the final locked-test DNN and machine-learning configurations for the binary, three-class, and five-class tasks.")
    print("=" * 92)


if __name__ == "__main__":
    main()
