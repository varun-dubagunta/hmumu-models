"""
model.py
--------
Custom Keras layers, loss functions, and model builder used by both
train.py and evaluate.py.
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention,
    Input, Concatenate, Reshape, GlobalAveragePooling1D, Layer,
)
from tensorflow.keras.models import Model

from config import (
    TOKEN_GROUPS, EMBEDDING_DIM, NUM_HEADS, DROPOUT_RATE,
    LEARNING_RATE, WEIGHT_DECAY, FOCAL_GAMMA, FOCAL_ALPHA,
)


# ── Custom Gather Layer ───────────────────────────────────────────────────────

class GatherLayer(Layer):
    """Selects a subset of feature columns by index (avoids Lambda serialisation issues)."""

    def __init__(self, indices=None, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def call(self, inputs):
        if self.indices is None:
            raise ValueError("GatherLayer not initialised with indices.")
        return tf.gather(inputs, self.indices, axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"indices": self.indices})
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ── Focal Loss ────────────────────────────────────────────────────────────────

def focal_loss(gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
    """
    Sparse-categorical focal loss.

    Parameters
    ----------
    gamma : float  Focus parameter – down-weights easy examples.
    alpha : float  Class-balance weight applied to the true class.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        modulating = tf.pow(1.0 - p_t, gamma)
        alpha_t = y_true_one_hot * alpha + (1 - y_true_one_hot) * (1 - alpha)

        loss = modulating[:, None] * alpha_t * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return focal_loss_fixed


# ── Model builder ─────────────────────────────────────────────────────────────

def build_transformer(
    feature_columns,
    num_classes: int,
    embedding_dim: int = EMBEDDING_DIM,
    num_heads: int = NUM_HEADS,
    dropout: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    use_focal_loss: bool = True,
) -> Model:
    """
    Build and compile the physics-token Transformer model.

    Parameters
    ----------
    feature_columns : list[str]  Ordered list of input feature names.
    num_classes     : int        Number of output classes.

    Returns
    -------
    tf.keras.Model  Compiled model ready for training.
    """
    n_features = len(feature_columns)
    input_layer = Input(shape=(n_features,), name="input")

    # Build per-token projections
    token_list = []
    for i, (group_name, col_names) in enumerate(TOKEN_GROUPS.items()):
        indices = [feature_columns.index(c) for c in col_names]
        token = GatherLayer(indices, name=f"gather_{group_name}")(input_layer)
        token = Dense(embedding_dim, name=f"embed_{group_name}")(token)
        token = Reshape((1, embedding_dim))(token)
        token_list.append(token)

    tokens = Concatenate(axis=1)(token_list)
    tokens = Dropout(dropout)(tokens)

    # Transformer encoder blocks
    for block in range(2):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout,
            name=f"mha_{block}",
        )(tokens, tokens)
        tokens = LayerNormalization(name=f"ln1_{block}")(tokens + attn)

        ff = Dense(embedding_dim * 4, activation="gelu", name=f"ff1_{block}")(tokens)
        ff = Dropout(dropout, name=f"drop_ff_{block}")(ff)
        ff = Dense(embedding_dim, name=f"ff2_{block}")(ff)
        tokens = LayerNormalization(name=f"ln2_{block}")(tokens + ff)

    x = GlobalAveragePooling1D()(tokens)
    x = Dense(128, activation="gelu")(x)
    x = LayerNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="gelu")(x)
    x = LayerNormalization()(x)
    x = Dropout(0.10)(x)

    output = Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inputs=input_layer, outputs=output, name="HmumuTransformer")

    loss_fn = focal_loss() if use_focal_loss else "sparse_categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def load_trained_model(path):
    """Load a saved .keras model with all required custom objects."""
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "GatherLayer": GatherLayer,
            "focal_loss_fixed": focal_loss(),
        },
    )