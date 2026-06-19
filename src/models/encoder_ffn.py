"""Encoder-FFN architecture and ablation variants.

Three model builders are exposed:
    build_encoder_ffn(...)          – full dual-branch model (paper)
    build_ffn_only(...)             – numerical branch only  (ablation)
    build_encoder_only(...)         – textual branch only    (ablation)

All three share the same final dense head and compilation so results are
directly comparable.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .. import config


def _compile(model: keras.Model, lr: float = config.LR_INITIAL) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def _dense_head(x, dropout_rate: float, dropout_mid: float, l2_reg: float):
    """Shared classification head: Dropout → Dense(64) → Dropout → Dense(32) → sigmoid."""
    x = layers.Dropout(dropout_rate, name="dropout_fusion")(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense_final_1")(x)
    x = layers.Dropout(dropout_mid, name="dropout_mid")(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense_final_2")(x)
    return layers.Dense(1, activation="sigmoid", name="output")(x)


# ---------------------------------------------------------------------------
# Numerical branch builder
# ---------------------------------------------------------------------------
def _numerical_branch(n_num: int) -> tuple:
    """Returns (input_tensor, branch_output_tensor)."""
    inp = keras.Input(shape=(n_num,), name="input_numerical")
    x   = layers.Dense(config.FFN_HIDDEN, activation="relu", name="ffn_num_1")(inp)
    x   = layers.Dense(config.FFN_HIDDEN, activation="relu", name="ffn_num_2")(x)
    return inp, x


# ---------------------------------------------------------------------------
# Textual branch builder
# ---------------------------------------------------------------------------
def _textual_branch(vocab_size: int,
                    embedding_dim: int = config.EMBEDDING_DIM,
                    num_heads: int = config.NUM_HEADS,
                    key_dim: int = config.KEY_DIM,
                    max_seq: int = config.MAX_SEQ_LEN) -> tuple:
    """Returns (input_tensor, branch_output_tensor [dim=embedding_dim])."""
    inp   = keras.Input(shape=(max_seq,), name="input_textual")
    x_emb = layers.Embedding(vocab_size, embedding_dim, name="word_embedding")(inp)

    # Multi-Head Self-Attention + residual
    attn  = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="multi_head_attention"
    )(x_emb, x_emb)
    x_txt = layers.Add(name="residual_attn")([x_emb, attn])
    x_txt = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x_txt)

    # Position-wise FFN + residual
    ffn   = layers.Dense(embedding_dim, activation="relu", name="ffn_txt_inner")(x_txt)
    ffn   = layers.Dense(embedding_dim, name="ffn_txt_outer")(ffn)
    x_txt = layers.Add(name="residual_ffn")([x_txt, ffn])
    x_txt = layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x_txt)

    x_txt = layers.GlobalAveragePooling1D(name="global_avg_pool")(x_txt)
    return inp, x_txt


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_encoder_ffn(n_num_features: int, vocab_size: int,
                      dropout_rate: float = config.DROPOUT_RATE,
                      dropout_mid: float  = config.DROPOUT_MID,
                      l2_reg: float       = config.L2_REG,
                      lr: float           = config.LR_INITIAL,
                      name: str = "Encoder_FFN") -> keras.Model:
    """Full dual-branch Encoder-FFN model as described in the paper.

    Architecture:
      Numerical input → MinMax-scaled → FFN(32, ReLU) × 2  ─────────────┐
                                                                          ├─ Concat[96]
      Textual input  → Embedding(64) → MHA(4 heads) → LayerNorm          │
                     → FFN_txt → LayerNorm → GlobalAvgPool1D  ────────────┘
                     → Dropout(0.35) → Dense(64) → Dropout(0.20)
                     → Dense(32) → Dense(1, sigmoid)

    Tokenisation details (for reproducibility):
      - Keras Tokenizer with OOV token '<OOV>', trained from scratch on DNS data.
      - Vocabulary sizes: 800 (Marques), 2000 (BCCC) — chosen to cover the DNS
        token space without excessive sparsity.
      - Max sequence length: 16 tokens (covers > 99% of DNS response text fields).
      - Embedding dimension 64: empirically selected; larger values (128, 256) did
        not improve validation AUC on Marques but increased training time.
    """
    inp_num, h_num = _numerical_branch(n_num_features)
    inp_txt, h_txt = _textual_branch(vocab_size)

    fused = layers.Concatenate(name="concat_branches")([h_num, h_txt])
    output = _dense_head(fused, dropout_rate, dropout_mid, l2_reg)

    model = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)


def build_ffn_only(n_num_features: int,
                   dropout_rate: float = config.DROPOUT_RATE,
                   dropout_mid: float  = config.DROPOUT_MID,
                   l2_reg: float       = config.L2_REG,
                   lr: float           = config.LR_INITIAL,
                   name: str = "FFN_Only") -> keras.Model:
    """Ablation: numerical branch ONLY (no textual encoder).

    A dummy textual input is still accepted so the same training loop works,
    but it is not connected to the graph.
    """
    inp_num, h_num = _numerical_branch(n_num_features)
    # Dummy textual input (ignored)
    inp_txt = keras.Input(shape=(config.MAX_SEQ_LEN,), name="input_textual")

    output = _dense_head(h_num, dropout_rate, dropout_mid, l2_reg)
    model  = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)


def build_encoder_only(n_num_features: int, vocab_size: int,
                       dropout_rate: float = config.DROPOUT_RATE,
                       dropout_mid: float  = config.DROPOUT_MID,
                       l2_reg: float       = config.L2_REG,
                       lr: float           = config.LR_INITIAL,
                       name: str = "Encoder_Only") -> keras.Model:
    """Ablation: textual encoder branch ONLY (no numerical FFN).

    A dummy numerical input is still accepted so the same training loop works,
    but it is not connected to the graph.
    """
    # Dummy numerical input (ignored)
    inp_num = keras.Input(shape=(n_num_features,), name="input_numerical")
    inp_txt, h_txt = _textual_branch(vocab_size)

    output = _dense_head(h_txt, dropout_rate, dropout_mid, l2_reg)
    model  = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)
