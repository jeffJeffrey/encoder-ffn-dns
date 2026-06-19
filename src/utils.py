"""Shared utilities: figure saving (PDF + PNG), JSON I/O, latency measurement.

All plots in this project go through `save_fig` so they are consistently
exported as PDF (vector, for the paper) and PNG (quick preview).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


_IN_NOTEBOOK = _in_notebook()
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Figure saving — PDF (paper-quality) + PNG (quick preview)
# ---------------------------------------------------------------------------

def save_fig(fig: Figure, out_dir: Path, name: str, dpi: int = 200) -> Path:
    """Save *fig* as PDF + PNG under *out_dir/name*.{pdf,png}.

    PDF is the primary output (vector quality for the paper).
    PNG is kept for quick visual inspection.
    The figure is also shown inline when running in a Jupyter/Colab notebook.

    Returns the PDF path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    if _IN_NOTEBOOK:
        try:
            from IPython.display import display
            display(fig)
        except Exception:
            pass

    plt.close(fig)
    print(f"[saved] {pdf_path}")
    return pdf_path


def save_json(obj: dict, path: Path) -> None:
    """Persist *obj* as pretty-printed JSON at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
# Model efficiency
# ---------------------------------------------------------------------------

def count_params(model) -> dict:
    trainable     = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"trainable": trainable, "non_trainable": non_trainable,
            "total": trainable + non_trainable}


def measure_latency(model, input_shape, n_runs: int = 100, warmup: int = 10) -> dict:
    """Measure average single-sample inference latency (ms) on CPU/GPU."""
    import tensorflow as tf
    x = tf.random.normal([1, *input_shape])
    for _ in range(warmup):
        model(x, training=False)
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x, training=False)
    elapsed = (time.perf_counter() - start) / n_runs
    return {"latency_ms": round(elapsed * 1000.0, 4), "n_runs": n_runs}
