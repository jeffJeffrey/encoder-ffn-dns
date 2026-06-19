"""Encoder-FFN architecture and ablation variants.

Three model builders are exposed:
    build_encoder_ffn(...)   – full dual-branch model (paper)
    build_ffn_only(...)      – numerical branch only  (ablation)
    build_encoder_only(...)  – textual branch only    (ablation)

All three accept [inp_txt, inp_num] so the same training/eval loop works.
The ablation variants route only one branch to the head; the unused input
is consumed via a zero-multiply trick so Keras does not raise
"inputs not connected to outputs".
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow as tf

from .. import config


# ---------------------------------------------------------------------------
# Compilation helper
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Shared classification head
# ---------------------------------------------------------------------------
def _dense_head(x, dropout_rate: float, dropout_mid: float,
                l2_reg: float, suffix: str = "") -> keras.KerasTensor:
    """Dropout → Dense(64,ReLU) → Dropout → Dense(32,ReLU) → Dense(1,sigmoid)."""
    x = layers.Dropout(dropout_rate, name=f"dropout_fusion{suffix}")(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name=f"dense_final_1{suffix}")(x)
    x = layers.Dropout(dropout_mid, name=f"dropout_mid{suffix}")(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name=f"dense_final_2{suffix}")(x)
    return layers.Dense(1, activation="sigmoid", name=f"output{suffix}")(x)


# ---------------------------------------------------------------------------
# Branch builders  (each returns (input_tensor, branch_output_tensor))
# ---------------------------------------------------------------------------
def _numerical_branch(n_num: int, suffix: str = "") -> tuple:
    inp = keras.Input(shape=(n_num,), name="input_numerical")
    x   = layers.Dense(config.FFN_HIDDEN, activation="relu",
                       name=f"ffn_num_1{suffix}")(inp)
    x   = layers.Dense(config.FFN_HIDDEN, activation="relu",
                       name=f"ffn_num_2{suffix}")(x)
    return inp, x


def _textual_branch(vocab_size: int,
                    embedding_dim: int = config.EMBEDDING_DIM,
                    num_heads: int     = config.NUM_HEADS,
                    key_dim: int       = config.KEY_DIM,
                    max_seq: int       = config.MAX_SEQ_LEN,
                    suffix: str = "") -> tuple:
    inp   = keras.Input(shape=(max_seq,), name="input_textual")
    x_emb = layers.Embedding(vocab_size, embedding_dim,
                              name=f"word_embedding{suffix}")(inp)

    attn  = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim,
        name=f"multi_head_attention{suffix}"
    )(x_emb, x_emb)
    x_txt = layers.Add(name=f"residual_attn{suffix}")([x_emb, attn])
    x_txt = layers.LayerNormalization(epsilon=1e-6,
                                      name=f"layer_norm_1{suffix}")(x_txt)

    ffn   = layers.Dense(embedding_dim, activation="relu",
                         name=f"ffn_txt_inner{suffix}")(x_txt)
    ffn   = layers.Dense(embedding_dim,
                         name=f"ffn_txt_outer{suffix}")(ffn)
    x_txt = layers.Add(name=f"residual_ffn{suffix}")([x_txt, ffn])
    x_txt = layers.LayerNormalization(epsilon=1e-6,
                                      name=f"layer_norm_2{suffix}")(x_txt)
    x_txt = layers.GlobalAveragePooling1D(
                name=f"global_avg_pool{suffix}")(x_txt)
    return inp, x_txt


# ---------------------------------------------------------------------------
# "Dead" input connector — makes Keras accept an unused input in the graph
# ---------------------------------------------------------------------------
def _dead_connect(unused_inp: keras.KerasTensor,
                  live_out:   keras.KerasTensor,
                  name: str = "dead") -> keras.KerasTensor:
    """Add unused_inp * 0 to live_out so Keras sees it as connected.

    This is the standard trick to keep an input in the graph without
    affecting the computation: output = live_out + stop_gradient(inp * 0).
    """
    # Reduce unused_inp to a scalar-per-sample, multiply by zero, add to output
    dead = layers.Lambda(
        lambda x: tf.zeros_like(x[:, :1]),
        name=f"{name}_zero"
    )(unused_inp)
    # Tile to match live_out last dim
    live_dim = live_out.shape[-1]
    dead = layers.Lambda(
        lambda x: tf.tile(x, [1, live_dim]),
        name=f"{name}_tile"
    )(dead)
    return layers.Add(name=f"{name}_add")([live_out, dead])


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_encoder_ffn(n_num_features: int, vocab_size: int,
                      dropout_rate: float = config.DROPOUT_RATE,
                      dropout_mid: float  = config.DROPOUT_MID,
                      l2_reg: float       = config.L2_REG,
                      lr: float           = config.LR_INITIAL,
                      name: str = "Encoder_FFN") -> keras.Model:
    """Full dual-branch Encoder-FFN model (paper).

    Inputs:  [input_textual (seq,), input_numerical (n_num,)]
    Output:  P(attack) ∈ [0,1]

    Tokenisation:
      - Keras Tokenizer trained from scratch on DNS data (OOV token '<OOV>').
      - Vocab sizes: 800 (Marques), 2000 (BCCC).
      - Max sequence length: 16.
      - Embedding dim 64 (random init, trained end-to-end).
    """
    inp_num, h_num = _numerical_branch(n_num_features)
    inp_txt, h_txt = _textual_branch(vocab_size)

    fused  = layers.Concatenate(name="concat_branches")([h_num, h_txt])
    output = _dense_head(fused, dropout_rate, dropout_mid, l2_reg)

    model = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)


def build_ffn_only(n_num_features: int,
                   dropout_rate: float = config.DROPOUT_RATE,
                   dropout_mid: float  = config.DROPOUT_MID,
                   l2_reg: float       = config.L2_REG,
                   lr: float           = config.LR_INITIAL,
                   name: str = "FFN_Only") -> keras.Model:
    """Ablation — numerical branch ONLY.

    The textual input is still declared (same signature as full model)
    but contributes zero to the computation via a dead-connect.
    Inputs:  [input_textual (seq,), input_numerical (n_num,)]
    """
    inp_num, h_num = _numerical_branch(n_num_features)
    inp_txt = keras.Input(shape=(config.MAX_SEQ_LEN,), name="input_textual")

    # Connect inp_txt to graph with zero contribution
    h_num_conn = _dead_connect(inp_txt, h_num, name="txt_dead")
    output = _dense_head(h_num_conn, dropout_rate, dropout_mid, l2_reg)

    model = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)


def build_encoder_only(n_num_features: int, vocab_size: int,
                       dropout_rate: float = config.DROPOUT_RATE,
                       dropout_mid: float  = config.DROPOUT_MID,
                       l2_reg: float       = config.L2_REG,
                       lr: float           = config.LR_INITIAL,
                       name: str = "Encoder_Only") -> keras.Model:
    """Ablation — textual encoder ONLY.

    The numerical input is still declared (same signature as full model)
    but contributes zero to the computation via a dead-connect.
    Inputs:  [input_textual (seq,), input_numerical (n_num,)]
    """
    inp_txt, h_txt = _textual_branch(vocab_size)
    inp_num = keras.Input(shape=(n_num_features,), name="input_numerical")

    # Connect inp_num to graph with zero contribution
    h_txt_conn = _dead_connect(inp_num, h_txt, name="num_dead")
    output = _dense_head(h_txt_conn, dropout_rate, dropout_mid, l2_reg)

    model = keras.Model(inputs=[inp_txt, inp_num], outputs=output, name=name)
    return _compile(model, lr)