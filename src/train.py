"""
train.py
--------
Load raw data → preprocess → train a selected model → save model + scaler.

Run:
    python src/train.py --model transformer_focal   (default)
    python src/train.py --model transformer_ce
    python src/train.py --model dnn
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)

# Make sure sibling modules are importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BATCH_SIZE, DATA_COLUMNS, DATA_DIR,
    EPOCHS, MODEL_DIR, PREDICTIONS_FILE,
    RANDOM_STATE, SIGNAL_PROCESSES, TEST_SIZE,
)

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (builder_fn, uses_sample_weights, best_filename, final_filename, scaler_filename)
def _get_model_config(model_key: str):
    if model_key == "transformer_focal":
        from model import build_transformer
        return dict(
            build_fn=lambda cols, n: build_transformer(cols, n, use_focal_loss=True),
            sample_weights=False,   # focal loss handles imbalance internally
            best_file="best_transformer_focal.keras",
            final_file="final_transformer_focal.keras",
            scaler_file="scaler_transformer_focal.pkl",
        )
    elif model_key == "transformer_ce":
        from model_transformer_ce import build_transformer_ce
        return dict(
            build_fn=build_transformer_ce,
            sample_weights=True,
            best_file="best_transformer_ce.keras",
            final_file="final_transformer_ce.keras",
            scaler_file="scaler_transformer_ce.pkl",
        )
    elif model_key == "dnn":
        from model_dnn import build_dnn
        return dict(
            build_fn=lambda cols, n: build_dnn(n_features=len(cols), num_classes=n),
            sample_weights=True,
            best_file="best_dnn.keras",
            final_file="final_dnn.keras",
            scaler_file="scaler_dnn.pkl",
        )
    elif model_key == "dnn_focal":
        from model_dnn import build_dnn
        return dict(
            build_fn=lambda cols, n: build_dnn(n_features=len(cols), num_classes=n,
                                               use_focal_loss=True),
            sample_weights=False,  # focal loss handles imbalance internally
            best_file="best_dnn_focal.keras",
            final_file="final_dnn_focal.keras",
            scaler_file="scaler_dnn_focal.pkl",
        )
    else:
        raise ValueError(f"Unknown model '{model_key}'. Choose: transformer_focal, transformer_ce, dnn")


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_data(path):
    print(f"Loading data from {path} …")
    df = pd.read_pickle(path)
    print(f"  {len(df):,} rows  |  {df.shape[1]} columns")
    return df


# ── 2. Preprocess ─────────────────────────────────────────────────────────────

def preprocess(df):
    meta_cols = [c for c in df.columns if c not in DATA_COLUMNS]

    X = df[DATA_COLUMNS].copy()
    y = df[meta_cols].copy()

    # Merge process names before encoding (e.g. combine all DY subtypes)
    from config import PROCESS_REMAP
    if PROCESS_REMAP:
        y["process"] = y["process"].replace(PROCESS_REMAP)
        print(f"  Process remapping applied: {PROCESS_REMAP}")

    # Encode process labels
    le = LabelEncoder()
    y["Label"] = le.fit_transform(y["process"])
    X["Label"] = y["Label"]

    print(f"  Classes: {list(le.classes_)}")
    return X, y, le


def compute_balanced_sample_weights(y_true, raw_weights, target_yield=25000.0):
    """
    Transforms raw physical weights into balanced training weights.
    1. Scales the sum of weights for each class to equal target_yield.
    2. Takes abs value to neutralise negative MC weights during loss calculation.
    """
    y_true      = np.asarray(y_true)
    raw_weights = np.asarray(raw_weights)
    training_weights = np.zeros_like(raw_weights, dtype=np.float32)

    for cls in np.unique(y_true):
        mask        = (y_true == cls)
        class_yield = np.sum(raw_weights[mask])
        scale_factor = target_yield / class_yield if class_yield != 0 else 0.0
        training_weights[mask] = np.abs(raw_weights[mask] * scale_factor)

    return training_weights
def compute_inverse_frequency_weights(y_true):
    """
    Per-event weight = N_total / (N_classes * N_class_i)
    Ignores physics weights entirely — purely based on MC event counts.
    """
    y_true = np.asarray(y_true)
    classes, counts = np.unique(y_true, return_counts=True)
    n_total = len(y_true)
    n_classes = len(classes)
    weights = np.zeros(len(y_true), dtype=np.float32)
    for cls, count in zip(classes, counts):
        mask = y_true == cls
        weights[mask] = n_total / (n_classes * count)
    return weights

def split_and_scale(X, y):
    X_features = X.drop(columns=["Label"])

    from config import USE_INVERSE_FREQUENCY, CLASS_TARGET_YIELDS, PROCESS_REMAP

    if USE_INVERSE_FREQUENCY:
        weights = pd.Series(
            compute_inverse_frequency_weights(y["Label"].values),
            index=y.index,
        )
        print("  Using inverse frequency weights (no physics weights).")

    elif CLASS_TARGET_YIELDS and "process" in y.columns:
        if "Class_Weight" in y.columns:
            raw_weights = y["Class_Weight"].values
            print("  Using 'Class_Weight' column as raw physical weights.")
        else:
            raw_weights = np.ones(len(y), dtype=np.float32)
            print("  No 'Class_Weight' column found – using uniform raw weights.")

        proc_col = y["process"].replace(PROCESS_REMAP)
        training_weights = np.zeros(len(y), dtype=np.float32)
        label_to_yield = {}
        for lbl in np.unique(y["Label"].values):
            proc_name = proc_col[y["Label"].values == lbl].iloc[0]
            label_to_yield[int(lbl)] = CLASS_TARGET_YIELDS.get(proc_name, 25000.0)
        for lbl, tgt in label_to_yield.items():
            mask        = y["Label"].values == lbl
            class_yield = np.sum(raw_weights[mask])
            scale       = tgt / class_yield if class_yield != 0 else 0.0
            training_weights[mask] = np.abs(raw_weights[mask] * scale)
        weights = pd.Series(training_weights, index=y.index)
        print("  Per-class target yields:")
        for lbl, tgt in sorted(label_to_yield.items()):
            proc = proc_col[y["Label"].values == lbl].iloc[0]
            w_sum = training_weights[y["Label"].values == lbl].sum()
            print(f"    {proc:<35} target={tgt:>6.0f}  effective={w_sum:.0f}")

    else:
        if "Class_Weight" in y.columns:
            raw_weights = y["Class_Weight"].values
        else:
            raw_weights = np.ones(len(y), dtype=np.float32)
        weights = pd.Series(
            compute_balanced_sample_weights(y["Label"].values, raw_weights),
            index=y.index,
        )
        print(f"  Balanced sample weights: target_yield=25000 per class, abs(w) applied.")

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_features, y, weights,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y["Label"],
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    mask  = np.any(np.abs(X_tr_sc) > 5, axis=1)
    n_out = mask.sum()
    print(f"  Removing {n_out} outlier rows ({n_out / len(X_tr) * 100:.2f}%)")
    X_tr_sc = X_tr_sc[~mask]
    y_tr    = y_tr[~mask]
    w_tr    = w_tr[~mask]

    X_val_sc = scaler.transform(X_val)

    return X_tr_sc, X_val_sc, y_tr, y_val, w_tr, w_val, scaler, X_features.columns.tolist()
# ── 3. Train ──────────────────────────────────────────────────────────────────

def train(X_tr, X_val, y_tr, y_val, w_tr, w_val, feature_columns, num_classes, model_cfg):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_path  = str(MODEL_DIR / model_cfg["best_file"])
    final_path = str(MODEL_DIR / model_cfg["final_file"])

    model = model_cfg["build_fn"](feature_columns, num_classes)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            best_path, monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=10,
            verbose=1, mode="max", min_lr=1e-5,
        ),
        EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    if model_cfg["sample_weights"]:
        fit_kwargs["sample_weight"] = w_tr
        fit_kwargs["validation_data"] = (X_val, y_val["Label"].astype(int), w_val)
    else:
        fit_kwargs["validation_data"] = (X_val, y_val["Label"].astype(int))

    history = model.fit(X_tr, y_tr["Label"].astype(int), **fit_kwargs)

    loss, acc = model.evaluate(X_val, y_val["Label"].astype(int), verbose=0)
    print(f"\nFinal validation  →  loss: {loss:.4f}  |  accuracy: {acc:.4f}")

    model.save(final_path)
    print(f"Model saved → {final_path}")

    return model, history


# ── 4. Generate + save predictions ───────────────────────────────────────────

def save_predictions(model, X_val_sc, y_val, y_full, le, X_all_sc, model_key, dataset_key):
    """
    Run inference on the full dataset and write predictions into
    data/<dataset_key>/<model_key>/ so evaluate.py can find them.
    """
    pred_dir = DATA_DIR / dataset_key / model_key
    pred_dir.mkdir(parents=True, exist_ok=True)

    preds_all = model.predict(X_all_sc, verbose=0)
    trained_labels = le.classes_

    sig_idx = [
        np.where(trained_labels == p)[0][0]
        for p in SIGNAL_PROCESSES if p in trained_labels
    ]
    bkg_idx = [i for i in range(len(trained_labels)) if i not in sig_idx]

    sig_prob = preds_all[:, sig_idx].sum(axis=1)
    bkg_prob = preds_all[:, bkg_idx].sum(axis=1)

    y_out = y_full.copy()
    y_out["predictions"] = sig_prob - bkg_prob

    preds_df = pd.DataFrame(preds_all, columns=trained_labels, index=y_full.index)
    preds_df["Actual_Process"] = y_full["process"].values
    if "Class_Weight" in y_full.columns:
        preds_df["Class_Weight"] = y_full["Class_Weight"].values

    y_out.to_pickle(str(pred_dir / "predictions_with_metadata.pkl"))
    preds_df.to_pickle(str(pred_dir / "predictions_df.pkl"))
    print(f"Predictions saved → {pred_dir}/")

    return y_out, preds_df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train an H→μμ classifier.")
    parser.add_argument(
        "--model",
        default="transformer_focal",
        choices=["transformer_focal", "transformer_ce", "dnn", "dnn_focal", "dnn_aux", "transformer_supcon", "gat", "combination"],
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3],
        help="For --model combination: start from phase 1 (default), 2 (skip branch pretraining), or 3 (skip to fine-tuning)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Short name for this dataset, used to namespace all outputs.\n"
            "e.g. --dataset v4 will read data/DNN_samples_v4.pkl and write\n"
            "to data/v4/<model>/ and models/v4/<model>/.\n"
            "Defaults to the stem of the data file (e.g. 'DNN_samples_v4')."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to input .pkl file. Defaults to data/DNN_samples_v4.pkl from config.",
    )
    args = parser.parse_args()

    from config import RAW_DATA_FILE
    data_path   = Path(args.data) if args.data else RAW_DATA_FILE
    dataset_key = args.dataset if args.dataset else data_path.stem

    run_model_dir = MODEL_DIR / dataset_key
    run_model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    X, y, le = preprocess(df)
    num_classes = int(y["Label"].nunique())

    # ── combination uses its own training loop ───────────────────────────────
    if args.model == "combination":
        from model_combination import train_combination
        print(f"\n{'='*60}")
        print(f"  Training model : {args.model}")
        print(f"  Dataset        : {dataset_key}  ({data_path.name})")
        print(f"{'='*60}\n")

        _, _, _, _, _, _, scaler, feature_columns = split_and_scale(X, y)
        print(f"  {num_classes} classes  |  {len(feature_columns)} features")

        model, _, scaler = train_combination(
            X, y, feature_columns, num_classes,
            save_dir=run_model_dir, dataset_key=dataset_key,
            start_phase=args.phase,
        )

        scaler_path = run_model_dir / "scaler_combination.pkl"
        with open(str(scaler_path), "wb") as f:
            pickle.dump({"scaler": scaler, "feature_columns": feature_columns, "label_encoder": le}, f)
        print(f"Scaler saved → {scaler_path}")

        X_features_all = X.drop(columns=["Label"])
        X_all_sc = scaler.transform(X_features_all)
        save_predictions(model, None, y, y, le, X_all_sc, args.model, dataset_key)
        print("\nTraining complete.")
        return

    # ── dnn_aux uses its own training loop ───────────────────────────────────
    if args.model == "dnn_aux":
        from model_dnn_aux import train_dnn_aux
        print(f"\n{'='*60}")
        print(f"  Training model : {args.model}")
        print(f"  Dataset        : {dataset_key}  ({data_path.name})")
        print(f"{'='*60}\n")

        _, _, _, _, _, _, scaler, feature_columns = split_and_scale(X, y)
        print(f"  {num_classes} classes  |  {len(feature_columns)} features")

        model, history, scaler = train_dnn_aux(
            X, y, feature_columns, num_classes,
            save_dir=run_model_dir, dataset_key=dataset_key,
        )

        scaler_path = run_model_dir / "scaler_dnn_aux.pkl"
        with open(str(scaler_path), "wb") as f:
            pickle.dump({"scaler": scaler, "feature_columns": feature_columns, "label_encoder": le}, f)
        print(f"Scaler saved → {scaler_path}")

        X_features_all = X.drop(columns=["Label"])
        X_all_sc = scaler.transform(X_features_all)
        save_predictions(model, None, y, y, le, X_all_sc, args.model, dataset_key)
        print("\nTraining complete.")
        return

    # ── GAT uses its own training loop ───────────────────────────────────────
    if args.model == "gat":
        from model_gat import train_gat
        print(f"\n{'='*60}")
        print(f"  Training model : {args.model}")
        print(f"  Dataset        : {dataset_key}  ({data_path.name})")
        print(f"{'='*60}\n")

        X_tr_sc, X_val_sc, y_tr, y_val, w_tr, w_val, scaler, feature_columns = split_and_scale(X, y)
        print(f"  {num_classes} classes  |  {len(feature_columns)} features")

        model, history, scaler = train_gat(
            X, y, feature_columns, num_classes,
            save_dir=run_model_dir, dataset_key=dataset_key,
        )

        scaler_path = run_model_dir / "scaler_gat.pkl"
        with open(str(scaler_path), "wb") as f:
            pickle.dump({"scaler": scaler, "feature_columns": feature_columns, "label_encoder": le}, f)
        print(f"Scaler saved → {scaler_path}")

        from model_gat import build_graph_tensors
        X_features_all = X.drop(columns=["Label"])
        X_all_df = pd.DataFrame(
            scaler.transform(X_features_all),
            columns=feature_columns, index=X_features_all.index,
        )
        node_all, _, edge_all, mass_all, nn_all = build_graph_tensors(X_all_df, feature_columns)
        preds_all = model.predict((node_all, edge_all, mass_all, nn_all), verbose=0)

        trained_labels = le.classes_
        sig_idx = [np.where(trained_labels == p)[0][0]
                   for p in SIGNAL_PROCESSES if p in trained_labels]
        bkg_idx = [i for i in range(len(trained_labels)) if i not in sig_idx]

        y_out = y.copy()
        y_out["predictions"] = preds_all[:, sig_idx].sum(axis=1) - preds_all[:, bkg_idx].sum(axis=1)
        y_out["final_weight"] = np.abs(y_out["Class_Weight"].values) if "Class_Weight" in y_out.columns \
                                else np.ones(len(y_out), dtype=np.float32)

        pred_dir = DATA_DIR / dataset_key / "gat"
        pred_dir.mkdir(parents=True, exist_ok=True)
        y_out.to_pickle(str(pred_dir / "predictions_with_metadata.pkl"))

        preds_df = pd.DataFrame(preds_all, columns=trained_labels, index=y.index)
        preds_df["Actual_Process"] = y["process"].values
        if "Class_Weight" in y.columns:
            preds_df["Class_Weight"] = y["Class_Weight"].values
        preds_df.to_pickle(str(pred_dir / "predictions_df.pkl"))
        print(f"Predictions saved → {pred_dir}/")
        print("\nTraining complete.")
        return
    if args.model == "transformer_supcon":
        from model_supcon import train_supcon
        print(f"\n{'='*60}")
        print(f"  Training model : {args.model}")
        print(f"  Dataset        : {dataset_key}  ({data_path.name})")
        print(f"{'='*60}\n")

        X_tr_sc, X_val_sc, y_tr, y_val, w_tr, w_val, scaler, feature_columns = split_and_scale(X, y)
        print(f"  {num_classes} classes  |  {len(feature_columns)} features")

        model, history, scaler = train_supcon(
            X, y, feature_columns, num_classes,
            save_dir=run_model_dir, dataset_key=dataset_key,
        )

        scaler_path = run_model_dir / "scaler_transformer_supcon.pkl"
        with open(str(scaler_path), "wb") as f:
            pickle.dump({"scaler": scaler, "feature_columns": feature_columns, "label_encoder": le}, f)
        print(f"Scaler saved → {scaler_path}")

        X_features_all = X.drop(columns=["Label"])
        X_all_sc = scaler.transform(X_features_all)
        save_predictions(model, X_val_sc, y_val, y, le, X_all_sc, args.model, dataset_key)
        print("\nTraining complete.")
        return

    # ── Standard training path ────────────────────────────────────────────────
    model_cfg = _get_model_config(args.model)
    print(f"\n{'='*60}")
    print(f"  Training model : {args.model}")
    print(f"  Dataset        : {dataset_key}  ({data_path.name})")
    print(f"  Best weights   : {run_model_dir}/{model_cfg['best_file']}")
    print(f"{'='*60}\n")

    model_cfg["best_file"]   = str(run_model_dir / model_cfg["best_file"])
    model_cfg["final_file"]  = str(run_model_dir / model_cfg["final_file"])
    model_cfg["scaler_file"] = str(run_model_dir / model_cfg["scaler_file"])

    X_tr_sc, X_val_sc, y_tr, y_val, w_tr, w_val, scaler, feature_columns = split_and_scale(X, y)
    print(f"  {num_classes} classes  |  {len(feature_columns)} features")

    model, history = train(
        X_tr_sc, X_val_sc, y_tr, y_val, w_tr, w_val,
        feature_columns, num_classes, model_cfg,
    )

    with open(model_cfg["scaler_file"], "wb") as f:
        pickle.dump({"scaler": scaler, "feature_columns": feature_columns, "label_encoder": le}, f)
    print(f"Scaler saved → {model_cfg['scaler_file']}")

    X_features_all = X.drop(columns=["Label"])
    X_all_sc = scaler.transform(X_features_all)
    save_predictions(model, X_val_sc, y_val, y, le, X_all_sc, args.model, dataset_key)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()