"""Training loop with callbacks.

Exposes:
    get_callbacks(name)
    train_model(model, data, name, class_weight=None, lr_override=None)
"""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

from . import config


def get_callbacks(name: str, model_dir=None):
    """Standard callbacks: EarlyStopping + ModelCheckpoint + ReduceLROnPlateau + TerminateOnNaN."""
    if model_dir is None:
        model_dir = config.MODELS_DIR
    os.makedirs(model_dir, exist_ok=True)
    ckpt_path = str(model_dir / f"{name}_best.keras")
    return [
        EarlyStopping(monitor="val_loss", patience=config.ES_PATIENCE,
                      min_delta=config.ES_MIN_DELTA, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def train_model(model, data: dict, name: str = "model",
                class_weight=None, lr_override: float | None = None):
    """Fit the model on data['X_tr_*'] and validate on data['X_val_*'].

    Args:
        model: compiled Keras model.
        data:  dict produced by preprocessing.full_pipeline().
        name:  used for checkpoint file naming and logging.
        class_weight: optional dict for imbalanced datasets.
        lr_override: if not None, recompile with this LR before training.

    Returns:
        history: Keras History object.
    """
    if lr_override is not None:
        model.optimizer.learning_rate.assign(lr_override)

    callbacks = get_callbacks(name.lower().replace(" ", "_"))
    history = model.fit(
        x=[data["X_tr_seq"], data["X_tr_num"]],
        y=data["y_tr"],
        validation_data=([data["X_val_seq"], data["X_val_num"]], data["y_val"]),
        epochs=config.EPOCHS_MAX,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    best_epoch   = int(np.argmin(history.history["val_loss"])) + 1
    best_val_acc = history.history["val_accuracy"][best_epoch - 1]
    print(f"\nTraining done — best epoch: {best_epoch}  "
          f"val_loss: {min(history.history['val_loss']):.4f}  "
          f"val_acc: {best_val_acc:.4f}")
    return history
