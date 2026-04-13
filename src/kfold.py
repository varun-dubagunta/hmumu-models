"""
kfold.py

K-fold cross-validation with event-ID-based splitting for the H→μμ classifier.

Events are grouped by contiguous chunks of FullEventId so that all events
sharing an ID stay in the same fold — no leakage between train/val/test.

For each test fold i:
    - fold i: test  (held out, never seen during training)
    - fold (i+1) % k : val   (early stopping / LR scheduling)
    - remaining folds :  train

After all k folds complete, every event has a prediction from a model
that never trained on it. The per-fold predictions are concatenated into
a single combined file.

Usage

    # Standard training path with k-fold
    python src/kfold.py --model dnn_focal --dataset your_dataset --data your_data.pkl --kfold 5

"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ── KFolder ───────────────────────────────────────────────────────────────────

class KFolder:
    """
    Split a DataFrame into k folds based on chunks of a grouping column
    (default: FullEventId).

    Events sharing the same group key always land in the same fold,
    preventing data leakage during cross-validation.

    Parameters
    ----------
    k          : int   Number of folds (default: 5).
    group_col  : str   Column to group by (default: 'FullEventId').
    random_state : int Seed for reproducible shuffling of groups.
    """

    def __init__(self, k: int = 5, group_col: str = "FullEventId",
                 random_state: int = 42):
        self.k = k
        self.group_col = group_col
        self.random_state = random_state

    def split(self, df: pd.DataFrame):
        """
        Yield (test_indices, rest_indices) for each fold.

        Parameters
        ----------
        df : DataFrame  Must contain self.group_col.

        Yields
        ------
        (test_idx, rest_idx) : tuple of np.ndarray
            Integer positional indices into df.
        """
        if self.group_col not in df.columns:
            raise KeyError(
                f"Column '{self.group_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)[:20]}…"
            )

        # Get unique group keys and shuffle them deterministically
        unique_groups = df[self.group_col].unique()
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_groups)

        # Assign each unique group to a fold
        fold_assignments = np.array_split(unique_groups, self.k)

        for i in range(self.k):
            test_groups = set(fold_assignments[i])
            is_test = df[self.group_col].isin(test_groups).values

            test_idx = np.where(is_test)[0]
            rest_idx = np.where(~is_test)[0]

            yield test_idx, rest_idx

    def split_train_val_test(self, df: pd.DataFrame):
        """
        Yield (train_indices, val_indices, test_indices) for each fold.

        For fold i:
            - test  = fold i
            - val   = fold (i+1) % k
            - train = all remaining folds

        Yields
        ------
        (train_idx, val_idx, test_idx) : tuple of np.ndarray
        """
        if self.group_col not in df.columns:
            raise KeyError(
                f"Column '{self.group_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)[:20]}…"
            )

        unique_groups = df[self.group_col].unique()
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_groups)

        fold_assignments = np.array_split(unique_groups, self.k)

        for i in range(self.k):
            test_groups  = set(fold_assignments[i])
            val_groups   = set(fold_assignments[(i + 1) % self.k])
            train_groups = set()
            for j in range(self.k):
                if j != i and j != (i + 1) % self.k:
                    train_groups.update(fold_assignments[j])

            group_col_vals = df[self.group_col].values
            is_test  = np.isin(group_col_vals, list(test_groups))
            is_val   = np.isin(group_col_vals, list(val_groups))
            is_train = ~is_test & ~is_val

            yield (
                np.where(is_train)[0],
                np.where(is_val)[0],
                np.where(is_test)[0],
            )


# ── K-fold training orchestrator ─────────────────────────────────────────────

def run_kfold_training(
    X, y, le, feature_columns, num_classes,
    model_key, model_cfg, dataset_key,
    k=5, group_col="FullEventId",
):
    """

    Parameters

    X, y, le          : from preprocess()
    feature_columns   : list[str]
    num_classes       : int
    model_key         : str  e.g. 'dnn_focal'
    model_cfg         : dict from _get_model_config()
    dataset_key       : str  e.g. 'v2602c_samples_2024'
    k                 : int  number of folds
    group_col         : str  column to group-split on

    Returns
    combined_preds : DataFrame  All events with predictions from their
                                held-out fold.
    """
    from config import DATA_DIR, MODEL_DIR, SIGNAL_PROCESSES
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    )
    from config import BATCH_SIZE, EPOCHS

    run_model_dir = MODEL_DIR / dataset_key
    pred_dir      = DATA_DIR / dataset_key / model_key
    run_model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # We need FullEventId from the original df — it's in y (metadata cols)
    if group_col not in y.columns:
        raise KeyError(
            f"'{group_col}' not found in metadata columns. "
            f"Available: {list(y.columns)}"
        )

    kfolder = KFolder(k=k, group_col=group_col)
    X_features = X.drop(columns=["Label"])

    all_fold_preds = []
    all_fold_meta  = []
    fold_metrics   = []

    for fold_i, (train_idx, val_idx, test_idx) in enumerate(
        kfolder.split_train_val_test(y)
    ):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_i + 1}/{k}")
        print(f"  Train: {len(train_idx):,}  |  Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")
        print(f"{'='*60}\n")

        # ── Slice data ────────────────────────────────────────────────────────
        X_tr  = X_features.iloc[train_idx]
        X_val = X_features.iloc[val_idx]
        X_te  = X_features.iloc[test_idx]

        y_tr  = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        y_te  = y.iloc[test_idx]

        # ── Compute sample weights (train only) ──────────────────────────────
        from train import compute_balanced_sample_weights
        from config import USE_INVERSE_FREQUENCY, CLASS_TARGET_YIELDS, PROCESS_REMAP
        from train import compute_inverse_frequency_weights

        if USE_INVERSE_FREQUENCY:
            w_tr = compute_inverse_frequency_weights(y_tr["Label"].values)
            w_val = compute_inverse_frequency_weights(y_val["Label"].values)
        elif CLASS_TARGET_YIELDS and "process" in y.columns:
            raw_w_tr = y_tr["Class_Weight"].values if "Class_Weight" in y_tr.columns \
                       else np.ones(len(y_tr), dtype=np.float32)
            raw_w_val = y_val["Class_Weight"].values if "Class_Weight" in y_val.columns \
                        else np.ones(len(y_val), dtype=np.float32)

            proc_col = y["process"].replace(PROCESS_REMAP) if PROCESS_REMAP else y["process"]

            # Build per-label yield targets
            label_to_yield = {}
            for lbl in np.unique(y["Label"].values):
                proc_name = proc_col[y["Label"].values == lbl].iloc[0]
                label_to_yield[int(lbl)] = CLASS_TARGET_YIELDS.get(proc_name, 25000.0)

            w_tr = np.zeros(len(y_tr), dtype=np.float32)
            for lbl, tgt in label_to_yield.items():
                mask = y_tr["Label"].values == lbl
                cyield = np.sum(raw_w_tr[mask])
                scale = tgt / cyield if cyield != 0 else 0.0
                w_tr[mask] = np.abs(raw_w_tr[mask] * scale)

            w_val = np.zeros(len(y_val), dtype=np.float32)
            for lbl, tgt in label_to_yield.items():
                mask = y_val["Label"].values == lbl
                cyield = np.sum(raw_w_val[mask])
                scale = tgt / cyield if cyield != 0 else 0.0
                w_val[mask] = np.abs(raw_w_val[mask] * scale)
        else:
            raw_w = y_tr["Class_Weight"].values if "Class_Weight" in y_tr.columns \
                    else np.ones(len(y_tr), dtype=np.float32)
            w_tr = compute_balanced_sample_weights(y_tr["Label"].values, raw_w)
            raw_w_v = y_val["Class_Weight"].values if "Class_Weight" in y_val.columns \
                      else np.ones(len(y_val), dtype=np.float32)
            w_val = compute_balanced_sample_weights(y_val["Label"].values, raw_w_v)

        # ── Scale (fit on train only) ─────────────────────────────────────────
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)

        # Remove outliers from train
        outlier_mask = np.any(np.abs(X_tr_sc) > 5, axis=1)
        n_out = outlier_mask.sum()
        print(f"  Removing {n_out} outlier rows ({n_out / len(X_tr) * 100:.2f}%)")
        X_tr_sc = X_tr_sc[~outlier_mask]
        y_tr    = y_tr[~outlier_mask]
        w_tr    = w_tr[~outlier_mask]

        X_val_sc = scaler.transform(X_val)
        X_te_sc  = scaler.transform(X_te)

        # ── Build model (fresh for each fold) ─────────────────────────────────
        model = model_cfg["build_fn"](feature_columns, num_classes)
        if fold_i == 0:
            model.summary()

        fold_best  = str(run_model_dir / f"best_{model_key}_fold{fold_i}.keras")
        fold_final = str(run_model_dir / f"final_{model_key}_fold{fold_i}.keras")

        callbacks = [
            ModelCheckpoint(
                fold_best, monitor="val_accuracy",
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
            fit_kwargs["validation_data"] = (X_val_sc, y_val["Label"].astype(int), w_val)
        else:
            fit_kwargs["validation_data"] = (X_val_sc, y_val["Label"].astype(int))

        history = model.fit(X_tr_sc, y_tr["Label"].astype(int), **fit_kwargs)

        loss, acc = model.evaluate(X_val_sc, y_val["Label"].astype(int), verbose=0)
        print(f"\n  Fold {fold_i + 1} validation → loss: {loss:.4f}  |  accuracy: {acc:.4f}")
        fold_metrics.append({"fold": fold_i, "val_loss": loss, "val_acc": acc})

        model.save(fold_final)

        # ── Save scaler for this fold ─────────────────────────────────────────
        scaler_path = run_model_dir / f"scaler_{model_key}_fold{fold_i}.pkl"
        with open(str(scaler_path), "wb") as f:
            pickle.dump({
                "scaler": scaler,
                "feature_columns": feature_columns,
                "label_encoder": le,
                "fold": fold_i,
                "k": k,
            }, f)

        # ── Predict on held-out test fold ─────────────────────────────────────
        preds_te = model.predict(X_te_sc, verbose=0)

        trained_labels = le.classes_
        sig_idx = [np.where(trained_labels == p)[0][0]
                   for p in SIGNAL_PROCESSES if p in trained_labels]
        bkg_idx = [i for i in range(len(trained_labels)) if i not in sig_idx]

        # Build predictions DataFrame for this fold's test set
        y_te_out = y_te.copy()
        y_te_out["predictions"] = (
            preds_te[:, sig_idx].sum(axis=1) - preds_te[:, bkg_idx].sum(axis=1)
        )
        y_te_out["fold"] = fold_i

        preds_te_df = pd.DataFrame(preds_te, columns=trained_labels, index=y_te.index)
        preds_te_df["Actual_Process"] = y_te["process"].values
        preds_te_df["fold"] = fold_i
        if "Class_Weight" in y_te.columns:
            preds_te_df["Class_Weight"] = y_te["Class_Weight"].values

        # Save per-fold
        fold_pred_dir = pred_dir / f"fold{fold_i}"
        fold_pred_dir.mkdir(parents=True, exist_ok=True)
        y_te_out.to_pickle(str(fold_pred_dir / "predictions_with_metadata.pkl"))
        preds_te_df.to_pickle(str(fold_pred_dir / "predictions_df.pkl"))
        print(f"  Fold {fold_i + 1} predictions saved → {fold_pred_dir}/")

        all_fold_meta.append(y_te_out)
        all_fold_preds.append(preds_te_df)

    # ── Combine all folds ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Combining {k}-fold predictions")
    print(f"{'='*60}\n")

    combined_meta  = pd.concat(all_fold_meta,  axis=0).sort_index()
    combined_preds = pd.concat(all_fold_preds, axis=0).sort_index()

    # Sanity check: every event should appear exactly once
    n_total    = len(X_features)
    n_combined = len(combined_meta)
    print(f"  Total events     : {n_total:,}")
    print(f"  Combined coverage: {n_combined:,}")
    if n_combined == n_total:
        print("  ✓ Every event predicted exactly once")
    else:
        print(f"  ⚠ Mismatch: {n_total - n_combined:,} events missing or duplicated")

    # Save combined
    combined_meta.to_pickle(str(pred_dir / "predictions_with_metadata.pkl"))
    combined_preds.to_pickle(str(pred_dir / "predictions_df.pkl"))
    print(f"  Combined predictions → {pred_dir}/")

    # Save fold assignment map (event → fold)
    fold_map = combined_meta[["fold"]].copy()
    if group_col in combined_meta.columns:
        fold_map[group_col] = combined_meta[group_col]
    fold_map.to_pickle(str(pred_dir / "fold_assignments.pkl"))
    print(f"  Fold assignments     → {pred_dir}/fold_assignments.pkl")

    # Print summary
    print(f"\n  Per-fold metrics:")
    for m in fold_metrics:
        print(f"    Fold {m['fold']+1}: val_loss={m['val_loss']:.4f}  val_acc={m['val_acc']:.4f}")
    avg_loss = np.mean([m["val_loss"] for m in fold_metrics])
    avg_acc  = np.mean([m["val_acc"]  for m in fold_metrics])
    std_acc  = np.std([m["val_acc"]   for m in fold_metrics])
    print(f"    Mean:   val_loss={avg_loss:.4f}  val_acc={avg_acc:.4f} ± {std_acc:.4f}")

    return combined_preds


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from config import DATA_DIR, MODEL_DIR, RAW_DATA_FILE

    parser = argparse.ArgumentParser(
        description="Run k-fold cross-validation for an H→μμ classifier.",
    )
    parser.add_argument(
        "--model", default="dnn_focal",
        choices=["transformer_focal", "transformer_ce", "dnn", "dnn_focal"],
        help="Model architecture (default: dnn_focal).",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument(
        "--group-col", default="FullEventId",
        help="Column to group-split on (default: FullEventId).",
    )
    args = parser.parse_args()

    from train import load_data, preprocess, _get_model_config
    from config import DATA_COLUMNS

    data_path   = Path(args.data) if args.data else RAW_DATA_FILE
    dataset_key = args.dataset if args.dataset else data_path.stem

    df = load_data(data_path)
    X, y, le = preprocess(df)
    num_classes     = int(y["Label"].nunique())
    feature_columns = [c for c in DATA_COLUMNS if c in X.columns and c != "Label"]

    model_cfg = _get_model_config(args.model)

    print(f"\n  K-fold CV: k={args.k}, model={args.model}, dataset={dataset_key}")
    print(f"  Group column: {args.group_col}")
    print(f"  {num_classes} classes  |  {len(feature_columns)} features\n")

    run_kfold_training(
        X, y, le, feature_columns, num_classes,
        model_key=args.model,
        model_cfg=model_cfg,
        dataset_key=dataset_key,
        k=args.k,
        group_col=args.group_col,
    )

    print("\nK-fold training complete.")