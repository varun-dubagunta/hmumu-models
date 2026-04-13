"""
model_dnn.py
------------

No focal loss – uses sparse_categorical_crossentropy with sample weights
to handle class imbalance.

Typical usage
-------------
    from model_dnn import build_dnn, train_dnn

    model, history, scaler = train_dnn(X, y, feature_columns)
    model.save("models/best_dnn.keras")
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR,
    RANDOM_STATE, TEST_SIZE,
)

# ── Architecture hyper-parameters (matching notebook) ─────────────────────────
_LAYER_SIZES  = [75, 75, 50, 50]   # nodes per hidden layer
_DO_BATCHNORM = True
_DROPOUT      = 0.05


# ── Builder ───────────────────────────────────────────────────────────────────

def build_dnn(
    n_features: int,
    num_classes: int,
    layer_sizes: list = None,
    do_batchnorm: bool = _DO_BATCHNORM,
    dropout: float = _DROPOUT,
    learning_rate: float = LEARNING_RATE,
    use_focal_loss: bool = False,
) -> Sequential:
    if layer_sizes is None:
        layer_sizes = _LAYER_SIZES

    model = Sequential(name="Hmumu_DNN")

    for i, units in enumerate(layer_sizes):
        if i == 0:
            model.add(Dense(units, input_shape=(n_features,)))
        else:
            model.add(Dense(units))
        if do_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation("relu"))
        if 0.0 < dropout < 1.0:
            model.add(Dropout(dropout))

    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    if use_focal_loss:
        from model import focal_loss
        loss_fn = focal_loss()
    else:
        loss_fn = "sparse_categorical_crossentropy"

    model.compile(
        loss=loss_fn,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


# ── Training helper ───────────────────────────────────────────────────────────

def train_dnn(
    X,
    y,
    feature_columns,
    save_dir=None,
    test_size: float = TEST_SIZE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    """
    Full training pipeline for the feedforward DNN:
      split → scale → remove outliers → build → fit → save.

    Parameters
    ----------
    X               : DataFrame  Features (with 'Label' column present).
    y               : DataFrame  Metadata (must have 'Label' and 'process').
    feature_columns : list[str]  Input feature names (no 'Label').
    save_dir        : Path-like   Where to write model/scaler files.
                                  Defaults to config.MODEL_DIR.
    Returns
    -------
    (model, history, scaler)
    """
    save_dir = Path(save_dir) if save_dir else MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_weights = y["Class_Weight"].values if "Class_Weight" in y.columns else np.ones(len(y), dtype=np.float32)
    from train import compute_balanced_sample_weights
    weights = pd.Series(
        compute_balanced_sample_weights(y["Label"].values, raw_weights),
        index=y.index,
    )

    X_features = X.drop(columns=["Label"])

    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_features, y, weights,
        test_size=test_size, random_state=RANDOM_STATE, stratify=y["Label"],
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)

    # Remove outliers (> 5 std) from training set only
    mask = np.any(np.abs(X_tr_sc) > 5, axis=1)
    n_out = mask.sum()
    print(f"  Removing {n_out} outlier rows ({n_out / len(X_tr) * 100:.2f}%)")
    X_tr_sc = X_tr_sc[~mask]
    y_tr     = y_tr[~mask]
    w_tr     = w_tr[~mask]

    X_val_sc = scaler.transform(X_val)

    # ── Build ─────────────────────────────────────────────────────────────────
    num_classes = int(y["Label"].nunique())
    model = build_dnn(n_features=X_tr_sc.shape[1], num_classes=num_classes)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_path = str(save_dir / "best_dnn.keras")
    callbacks = [
        ModelCheckpoint(
            best_path, monitor="val_accuracy",
            mode="max", save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=25,
            verbose=1, mode="max", min_lr=1e-5,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True,
        ),
    ]

    # ── Fit ───────────────────────────────────────────────────────────────────
    history = model.fit(
        X_tr_sc, y_tr["Label"].astype(int),
        sample_weight=w_tr,
        validation_data=(X_val_sc, y_val["Label"].astype(int), w_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(
        X_val_sc, y_val["Label"].astype(int), sample_weight=w_val, verbose=0,
    )
    print(f"\nDNN validation  →  loss: {loss:.4f}  |  accuracy: {acc:.4f}")

    # ── Save final model + scaler ──────────────────────────────────────────────
    final_path = str(save_dir / "final_dnn.keras")
    model.save(final_path)
    print(f"Model saved → {final_path}")

    scaler_path = save_dir / "scaler_dnn.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_columns": feature_columns}, f)
    print(f"Scaler saved → {scaler_path}")

    return model, history, scaler


# ── Stand-alone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    from config import DATA_COLUMNS, RAW_DATA_FILE

    print(f"Loading {RAW_DATA_FILE} …")
    df = pd.read_pickle(str(RAW_DATA_FILE))

    meta_cols = [c for c in df.columns if c not in DATA_COLUMNS]
    X = df[DATA_COLUMNS].copy()
    y = df[meta_cols].copy()

    le = LabelEncoder()
    y["Label"] = le.fit_transform(y["process"])
    X["Label"] = y["Label"]

    print(f"Classes: {list(le.classes_)}")

    model, history, scaler = train_dnn(X, y, feature_columns=DATA_COLUMNS)
    print("Done.")