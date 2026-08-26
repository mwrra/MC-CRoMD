# -*- coding: utf-8 -*-
"""
MC_CRoMD_Ablation_Pearson_ThreeClass_256.py
===========================================
Validation-only Pearson-correlation ablation for MC-CRoMD.

Variants:
1) Alpha_Only            : 9 Cronbach's Alpha features
2) Dispersion_Only       : 216 dispersion features
3) Combined_MC_CRoMD     : 225 Alpha + dispersion features
4) Pearson_Only          : 9 Pearson features (one per 256x256 patch)
5) Dispersion_Pearson    : 225 dispersion + Pearson features

Pearson descriptor per patch = arithmetic mean of the finite pairwise
correlations r(R,G), r(R,B), r(G,B). This gives one inter-channel feature
per patch, matching the dimensionality of Cronbach's Alpha.

Scientific safeguards:
- Three-Class, 256x256 only.
- Existing group-aware Train/Validation partitions only.
- Locked Test is never loaded or used.
- Pearson preprocessing is fitted on Train only (median imputation,
  1.5*IQR clipping, standard scaling) and then applied unchanged to Validation.
- Same DNN architecture/training settings and seeds 42,43,44 as the original
  ablation program.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")

import gc
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# USER SETTINGS
# =============================================================================
BASE_DIR = Path(r"G:\My Research About Lung Canser\INASS")

# Used only if ImagePath stored in Train/Validation is missing or invalid.
# Change this path if necessary to the root folder that contains LC25000 images.
LC25000_ROOT = Path(r"G:\My Research About Lung Canser\OldResult\lung_colon_image_set")

PATCH_SIZE = 256
SCENARIO = "Three_Class"
SOURCE_VARIANT = "StandardScaled"
TARGET_COLUMN = "label"
SEEDS = (42, 43, 44)
RUN_EXISTING_VARIANTS = True
USE_PEARSON_CACHE = True
IQR_FACTOR = 1.5
PEARSON_EPS = 1e-12

OUTPUT_DIR = BASE_DIR / "10-AblationStudy" / "Three_Class_256_Pearson"

EXPECTED_ALPHA_COUNT = 9
EXPECTED_DISPERSION_COUNT = 216
EXPECTED_COMBINED_COUNT = 225
EXPECTED_PEARSON_COUNT = 9

DENSE_UNITS = (128, 64, 32)
DROPOUT_RATES = (0.30, 0.20)
INITIAL_LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 5
MIN_LEARNING_RATE = 1e-6

METADATA_COLUMNS = {
    "ImgName", "FileName", "ImagePath", "ClassName", "label", "SourceGroupID",
    "PatchSize", "ImageWidth", "ImageHeight", "ImageSHA256", "Split", "split",
    "stem", "filename", "tissue", "group_id", "local_cluster_label",
    "LC25000_ClassName", "LC25000_FileName", "LC25000_Tissue",
    "Previous_SourceGroupID", "Previous_SourceGroupID_From_Raw",
    "_match_stem_LC25000", "LC25000_GroupID", "LC25000_LocalCluster",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =============================================================================
# HELPERS
# =============================================================================
def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def normalize_filename(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1].lower()


def find_existing_file(folder: Path, stem: str) -> Path:
    for candidate in (folder / f"{stem}.csv.gz", folder / f"{stem}.csv", folder / f"{stem}.xlsx"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find: " + str(folder / stem))


def read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".csv.gz") or path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_split(split_name: str) -> tuple[pd.DataFrame, Path]:
    if split_name not in {"Train", "Validation"}:
        raise ValueError("Only Train and Validation are permitted; Test is prohibited.")
    folder = BASE_DIR / "3-DataPreprocessing" / str(PATCH_SIZE) / SCENARIO / SOURCE_VARIANT
    path = find_existing_file(folder, split_name)
    df = read_table(path)
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"{split_name}: missing {TARGET_COLUMN}")
    return df, path


def get_feature_columns(train: pd.DataFrame, val: pd.DataFrame) -> list[str]:
    a = [c for c in train.columns if c not in METADATA_COLUMNS and c != TARGET_COLUMN]
    b = [c for c in val.columns if c not in METADATA_COLUMNS and c != TARGET_COLUMN]
    if a != b:
        raise RuntimeError("Train/Validation feature columns differ.")
    if len(a) != EXPECTED_COMBINED_COUNT:
        raise RuntimeError(f"Expected 225 MC-CRoMD features, found {len(a)}")
    return a


def split_existing_features(all_features: list[str]) -> tuple[list[str], list[str]]:
    alpha = [f for f in all_features if f.lower().startswith("cronbach_")]
    disp = [f for f in all_features if not f.lower().startswith("cronbach_")]
    if len(alpha) != EXPECTED_ALPHA_COUNT or len(disp) != EXPECTED_DISPERSION_COUNT:
        raise RuntimeError(f"Unexpected feature structure: Alpha={len(alpha)}, Dispersion={len(disp)}")
    return alpha, disp


def choose_sample_id_column(df: pd.DataFrame) -> str:
    for c in ("FileName", "ImgName", "LC25000_FileName"):
        if c in df.columns:
            return c
    raise KeyError("No filename column found.")

# =============================================================================
# IMAGE RESOLUTION + PEARSON EXTRACTION
# =============================================================================
def build_image_index(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"LC25000_ROOT not found:\n{root}")
    idx: dict[str, Path] = {}
    ambiguous: set[str] = set()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            key = p.name.lower()
            if key in idx and idx[key] != p:
                ambiguous.add(key)
            else:
                idx[key] = p
    for key in ambiguous:
        idx.pop(key, None)
    if not idx:
        raise RuntimeError(f"No images found under {root}")
    print(f"Indexed {len(idx):,} unique image filenames; ambiguous excluded={len(ambiguous):,}")
    return idx


def resolve_image_paths(df: pd.DataFrame, split_name: str) -> list[Path]:
    if "ImagePath" in df.columns:
        direct = [Path(str(v)) if not pd.isna(v) else Path("") for v in df["ImagePath"]]
        if direct and all(p.is_file() for p in direct):
            print(f"{split_name}: using stored ImagePath values.")
            return direct
    key_col = choose_sample_id_column(df)
    idx = build_image_index(LC25000_ROOT)
    paths, missing = [], []
    for v in df[key_col]:
        p = idx.get(normalize_filename(v))
        if p is None:
            missing.append(str(v)); paths.append(Path(""))
        else:
            paths.append(p)
    if missing:
        raise FileNotFoundError(f"{split_name}: unmatched images={len(missing)}; examples={missing[:10]}")
    print(f"{split_name}: resolved {len(paths):,} images by filename.")
    return paths


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xc = x - x.mean(); yc = y - y.mean()
    denom = np.sqrt(np.dot(xc, xc) * np.dot(yc, yc))
    if (not np.isfinite(denom)) or denom <= PEARSON_EPS:
        return float("nan")
    return float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


def compute_image_pearson_features(path: Path) -> tuple[np.ndarray, int, int]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if (h, w) != (768, 768):
        raise ValueError(f"Expected 768x768, got {w}x{h}: {path}")
    out, undefined_pairs, fully_undefined_patches = [], 0, 0
    for r0 in range(0, 768, PATCH_SIZE):
        for c0 in range(0, 768, PATCH_SIZE):
            patch = rgb[r0:r0+PATCH_SIZE, c0:c0+PATCH_SIZE]
            r = patch[:, :, 0].reshape(-1); g = patch[:, :, 1].reshape(-1); b = patch[:, :, 2].reshape(-1)
            vals = np.array([safe_pearson(r,g), safe_pearson(r,b), safe_pearson(g,b)], dtype=float)
            finite = np.isfinite(vals)
            undefined_pairs += int((~finite).sum())
            if finite.any():
                out.append(float(vals[finite].mean()))
            else:
                out.append(float("nan")); fully_undefined_patches += 1
    arr = np.asarray(out, dtype=np.float64)
    if arr.size != EXPECTED_PEARSON_COUNT:
        raise RuntimeError(f"Expected 9 patches, found {arr.size}")
    return arr, undefined_pairs, fully_undefined_patches


def pearson_feature_names() -> list[str]:
    return [f"PearsonMean_Patch_{i:02d}" for i in range(1, 10)]


def extract_pearson_matrix(df: pd.DataFrame, split_name: str) -> tuple[np.ndarray, pd.DataFrame]:
    paths = resolve_image_paths(df, split_name)
    X = np.full((len(df), 9), np.nan, dtype=np.float64)
    audit = []
    id_col = choose_sample_id_column(df)
    for i, p in enumerate(paths):
        vals, bad_pairs, bad_patches = compute_image_pearson_features(p)
        X[i] = vals
        audit.append({"Split": split_name, "Row": i, "Sample_ID": str(df.iloc[i][id_col]), "ImagePath": str(p),
                      "Undefined_Pair_Count": bad_pairs, "All_Pairs_Undefined_Patch_Count": bad_patches})
        if (i+1) % 250 == 0 or i+1 == len(df):
            print(f"{split_name}: Pearson {i+1:,}/{len(df):,}")
    return X, pd.DataFrame(audit)

# =============================================================================
# TRAIN-ONLY PEARSON PREPROCESSING
# =============================================================================
def preprocess_pearson(train_raw: np.ndarray, val_raw: np.ndarray):
    train = train_raw.astype(np.float64, copy=True); val = val_raw.astype(np.float64, copy=True)
    train[~np.isfinite(train)] = np.nan; val[~np.isfinite(val)] = np.nan
    med = np.nanmedian(train, axis=0)
    if np.isnan(med).any():
        raise RuntimeError("At least one Pearson feature is entirely missing in Train.")
    train_missing = np.isnan(train).sum(axis=0); val_missing = np.isnan(val).sum(axis=0)
    train = np.where(np.isnan(train), med[None,:], train); val = np.where(np.isnan(val), med[None,:], val)
    q1 = np.quantile(train, .25, axis=0); q3 = np.quantile(train, .75, axis=0); iqr = q3-q1
    lo = q1 - IQR_FACTOR*iqr; hi = q3 + IQR_FACTOR*iqr
    train_clip = ((train < lo) | (train > hi)).sum(axis=0); val_clip = ((val < lo) | (val > hi)).sum(axis=0)
    train = np.clip(train, lo[None,:], hi[None,:]); val = np.clip(val, lo[None,:], hi[None,:])
    mean = train.mean(axis=0); std = train.std(axis=0, ddof=0)
    if np.any((~np.isfinite(std)) | (std <= PEARSON_EPS)):
        raise RuntimeError("Zero/invalid Train std in Pearson features.")
    train_s = ((train-mean)/std).astype(np.float32); val_s = ((val-mean)/std).astype(np.float32)
    params = pd.DataFrame({"Feature": pearson_feature_names(), "Train_Median": med, "Train_Q1": q1, "Train_Q3": q3,
                           "Train_IQR": iqr, "Lower_Clip": lo, "Upper_Clip": hi, "Train_Mean_After_Clip": mean,
                           "Train_STD_After_Clip": std, "Train_Missing": train_missing, "Validation_Missing": val_missing,
                           "Train_Clipped": train_clip, "Validation_Clipped": val_clip})
    return train_s, val_s, params


def save_pearson_table(df: pd.DataFrame, X: np.ndarray, split_name: str, path: Path):
    out = pd.DataFrame(X, columns=pearson_feature_names())
    id_col = choose_sample_id_column(df)
    out.insert(0, "label", df[TARGET_COLUMN].to_numpy())
    out.insert(0, id_col, df[id_col].astype(str).to_numpy())
    out.insert(0, "Split", split_name)
    if "SourceGroupID" in df.columns:
        out.insert(2, "SourceGroupID", df["SourceGroupID"].astype(str).to_numpy())
    out.to_excel(path, index=False)


def load_or_create_pearson(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_cache = OUTPUT_DIR / "Pearson_Features_Train.xlsx"
    val_cache = OUTPUT_DIR / "Pearson_Features_Validation.xlsx"
    param_file = OUTPUT_DIR / "Pearson_Preprocessing_Parameters.xlsx"
    audit_file = OUTPUT_DIR / "Pearson_Extraction_Audit.xlsx"
    names = pearson_feature_names()
    if USE_PEARSON_CACHE and train_cache.exists() and val_cache.exists() and param_file.exists():
        print("Using cached Pearson features.")
        tr = pd.read_excel(train_cache); va = pd.read_excel(val_cache)
        Xtr = tr[names].to_numpy(dtype=np.float32); Xva = va[names].to_numpy(dtype=np.float32)
        if Xtr.shape[0] != len(train_df) or Xva.shape[0] != len(val_df):
            raise RuntimeError("Pearson cache row count mismatch.")
        return Xtr, Xva, {"cache_used": True}
    train_raw, audit_tr = extract_pearson_matrix(train_df, "Train")
    val_raw, audit_va = extract_pearson_matrix(val_df, "Validation")
    Xtr, Xva, params = preprocess_pearson(train_raw, val_raw)
    params.to_excel(param_file, index=False)
    pd.concat([audit_tr, audit_va], ignore_index=True).to_excel(audit_file, index=False)
    save_pearson_table(train_df, Xtr, "Train", train_cache)
    save_pearson_table(val_df, Xva, "Validation", val_cache)
    return Xtr, Xva, {"cache_used": False, "parameters": str(param_file), "audit": str(audit_file)}

# =============================================================================
# DNN
# =============================================================================
def build_dnn(input_dim: int, n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.20)(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="MC_CRoMD_Pearson_Ablation")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
                  loss="sparse_categorical_crossentropy",
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model


def calc_metrics(y_true, y_pred, proba, n_classes):
    out = {
        "Validation_Accuracy": float(accuracy_score(y_true,y_pred)),
        "Validation_Precision_Macro": float(precision_score(y_true,y_pred,average="macro",zero_division=0)),
        "Validation_Recall_Macro": float(recall_score(y_true,y_pred,average="macro",zero_division=0)),
        "Validation_F1_Macro": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "Validation_F1_Weighted": float(f1_score(y_true,y_pred,average="weighted",zero_division=0)),
    }
    try:
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        out["Validation_ROC_AUC_OVR_Macro"] = float(roc_auc_score(y_bin,proba,average="macro",multi_class="ovr"))
    except Exception:
        out["Validation_ROC_AUC_OVR_Macro"] = float("nan")
    return out


def run_one(variant_name: str, X_train: np.ndarray, X_val: np.ndarray, feature_names: list[str], seed: int,
            train_df: pd.DataFrame, val_df: pd.DataFrame):
    set_global_seed(seed); tf.keras.backend.clear_session(); gc.collect()
    enc = LabelEncoder(); y_train = enc.fit_transform(train_df[TARGET_COLUMN]); y_val = enc.transform(val_df[TARGET_COLUMN])
    if len(enc.classes_) != 3: raise RuntimeError(f"Expected 3 classes, found {enc.classes_}")
    run_dir = OUTPUT_DIR / variant_name / f"Seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "Best_Model.keras"
    model = build_dnn(X_train.shape[1], 3)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min", restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE,
                                         min_lr=MIN_LEARNING_RATE, mode="min", verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", mode="min", save_best_only=True, verbose=0),
        keras.callbacks.TerminateOnNaN(),
    ]
    t0 = time.perf_counter()
    hist = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=MAX_EPOCHS,batch_size=BATCH_SIZE,
                     shuffle=True,callbacks=callbacks,verbose=2)
    fit_sec = time.perf_counter()-t0
    best = keras.models.load_model(model_path)
    t0 = time.perf_counter(); proba = np.asarray(best.predict(X_val,verbose=0),dtype=float); pred_sec = time.perf_counter()-t0
    y_pred = np.argmax(proba,axis=1).astype(int)
    metrics = calc_metrics(y_val,y_pred,proba,3)
    val_losses = np.asarray(hist.history["val_loss"],dtype=float)
    best_epoch = int(np.nanargmin(val_losses))+1; best_val_loss = float(np.nanmin(val_losses))
    pd.DataFrame(hist.history).to_csv(run_dir/"Training_History.csv",index_label="Epoch")
    names = [str(x) for x in enc.classes_]
    cm = confusion_matrix(y_val,y_pred,labels=np.arange(3))
    pd.DataFrame(cm,index=[f"True_{x}" for x in names],columns=[f"Pred_{x}" for x in names]).to_excel(run_dir/"Validation_Confusion_Matrix.xlsx")
    pred_df = pd.DataFrame({"Row_Index":val_df.index.to_numpy(),"True_Encoded":y_val,"Predicted_Encoded":y_pred,
                            "True_Label":[names[i] for i in y_val],"Predicted_Label":[names[i] for i in y_pred]})
    id_col = choose_sample_id_column(val_df); pred_df.insert(1,id_col,val_df[id_col].astype(str).to_numpy())
    if "SourceGroupID" in val_df.columns: pred_df.insert(2,"SourceGroupID",val_df["SourceGroupID"].astype(str).to_numpy())
    for i,n in enumerate(names): pred_df[f"Probability_{n}"] = proba[:,i]
    pred_df.to_csv(run_dir/"Validation_Predictions.csv",index=False)
    pd.DataFrame({"Rank":np.arange(1,len(feature_names)+1),"Feature":feature_names}).to_excel(run_dir/"Ablation_Features.xlsx",index=False)
    lines=[]; best.summary(print_fn=lines.append); (run_dir/"Model_Summary.txt").write_text("\n".join(lines),encoding="utf-8")
    result = {"Variant":variant_name,"Seed":seed,"Feature_Count":len(feature_names),"Train_Rows":len(train_df),
              "Validation_Rows":len(val_df),"Epochs_Ran":len(hist.history["loss"]),"Best_Epoch_By_Val_Loss":best_epoch,
              "Best_Val_Loss":best_val_loss,"Fit_Seconds":fit_sec,"Predict_Seconds":pred_sec,"Parameter_Count":best.count_params()}
    result.update(metrics)
    del model,best,y_train,y_val,y_pred,proba; tf.keras.backend.clear_session(); gc.collect()
    return result

# =============================================================================
# MAIN
# =============================================================================
def main():
    start = time.perf_counter(); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*95); print("MC-CRoMD PEARSON ABLATION - THREE-CLASS / 256x256"); print("Locked Test is NEVER used."); print("="*95)
    train_df, train_path = load_split("Train"); val_df, val_path = load_split("Validation")
    print(f"Train rows={len(train_df):,} | Validation rows={len(val_df):,}")
    if "SourceGroupID" in train_df.columns and "SourceGroupID" in val_df.columns:
        overlap=set(train_df["SourceGroupID"].astype(str)) & set(val_df["SourceGroupID"].astype(str))
        if overlap: raise RuntimeError(f"Train/Validation group overlap={len(overlap)}")
        print("Train/Validation SourceGroupID overlap: 0")
    all_features = get_feature_columns(train_df,val_df); alpha,disp = split_existing_features(all_features)
    p_train,p_val,p_info = load_or_create_pearson(train_df,val_df); pnames=pearson_feature_names()
    variants={}
    if RUN_EXISTING_VARIANTS:
        variants["Alpha_Only"]=(train_df[alpha].to_numpy(np.float32),val_df[alpha].to_numpy(np.float32),alpha)
        variants["Dispersion_Only"]=(train_df[disp].to_numpy(np.float32),val_df[disp].to_numpy(np.float32),disp)
        variants["Combined_MC_CRoMD"]=(train_df[all_features].to_numpy(np.float32),val_df[all_features].to_numpy(np.float32),all_features)
    variants["Pearson_Only"]=(p_train,p_val,pnames)
    variants["Dispersion_Pearson"]=(np.concatenate([train_df[disp].to_numpy(np.float32),p_train],axis=1),
                                    np.concatenate([val_df[disp].to_numpy(np.float32),p_val],axis=1),disp+pnames)
    for name,(xt,xv,names) in variants.items():
        if not np.isfinite(xt).all() or not np.isfinite(xv).all(): raise RuntimeError(f"{name}: non-finite values")
        print(f"{name:22s}: {xt.shape[1]} features")
    defs=[]
    for name,(_,_,names) in variants.items():
        defs += [{"Variant":name,"Rank":i,"Feature":f} for i,f in enumerate(names,1)]
    pd.DataFrame(defs).to_excel(OUTPUT_DIR/"Ablation_Feature_Definitions.xlsx",index=False)
    results=[]
    for name,(xt,xv,names) in variants.items():
        for seed in SEEDS:
            print(f"\nRunning {name} | seed={seed}")
            r=run_one(name,xt,xv,names,seed,train_df,val_df); results.append(r)
            pd.DataFrame(results).to_excel(OUTPUT_DIR/"Ablation_All_Runs.xlsx",index=False)
            print(f"Accuracy={r['Validation_Accuracy']:.6f} | Macro-F1={r['Validation_F1_Macro']:.6f} | AUC={r['Validation_ROC_AUC_OVR_Macro']:.6f}")
    runs=pd.DataFrame(results)
    summary=(runs.groupby(["Variant","Feature_Count"],as_index=False)
             .agg(Seeds=("Seed","nunique"),Mean_Validation_Accuracy=("Validation_Accuracy","mean"),
                  SD_Validation_Accuracy=("Validation_Accuracy","std"),
                  Mean_Validation_Precision_Macro=("Validation_Precision_Macro","mean"),
                  Mean_Validation_Recall_Macro=("Validation_Recall_Macro","mean"),
                  Mean_Validation_F1_Macro=("Validation_F1_Macro","mean"),
                  SD_Validation_F1_Macro=("Validation_F1_Macro","std"),
                  Mean_Validation_ROC_AUC_OVR_Macro=("Validation_ROC_AUC_OVR_Macro","mean")))
    order={"Alpha_Only":1,"Dispersion_Only":2,"Combined_MC_CRoMD":3,"Pearson_Only":4,"Dispersion_Pearson":5}
    summary["_Order"]=summary["Variant"].map(order); summary=summary.sort_values("_Order").drop(columns="_Order")
    summary.to_excel(OUTPUT_DIR/"Ablation_Summary.xlsx",index=False)
    meta={"purpose":"Validation-only correlation-based ablation","scenario":SCENARIO,"patch_size":PATCH_SIZE,
          "train_path":str(train_path),"validation_path":str(val_path),
          "locked_test_policy":"Test is never loaded or used.","seeds":list(SEEDS),
          "pearson_definition":"Per patch: arithmetic mean of finite r(R,G), r(R,B), r(G,B); one Pearson feature per patch.",
          "pearson_preprocessing":"Train-only median imputation + 1.5*IQR clipping + standard scaling; applied unchanged to Validation.",
          "pearson_info":p_info,"software":{"python":platform.python_version(),"tensorflow":tf.__version__,"numpy":np.__version__,
          "pandas":pd.__version__,"scikit_learn":sklearn.__version__,"opencv":cv2.__version__},"elapsed_seconds":time.perf_counter()-start}
    json_dump(OUTPUT_DIR/"Ablation_Metadata.json",meta)
    print("\n"+"="*95); print("COMPLETED"); print(summary.to_string(index=False)); print("Locked Test was not used.")

if __name__ == "__main__":
    main()
