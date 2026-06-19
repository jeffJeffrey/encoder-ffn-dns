"""Evaluation: metrics, plots (PDF+PNG), ablation comparison table, statistical tests.

Main entry points:
    evaluate_model(model, data, name, dataset_tag, figures_dir)
    evaluate_per_category(model, data, categories, name, figures_dir)
    ablation_comparison(results_dict, dataset_tag, figures_dir)
    mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, auc,
                              precision_recall_curve)

from . import config
from .utils import save_fig, save_json


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, data: dict, name: str,
                   dataset_tag: str,
                   figures_dir: Path,
                   threshold: float = 0.5) -> dict:
    """Full evaluation: classification report + 4 plots (PDF+PNG).

    Plots saved:
        confusion_matrix.{pdf,png}
        roc_curve.{pdf,png}
        score_distribution.{pdf,png}
        threshold_sweep.{pdf,png}

    Returns:
        metrics dict with Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    """
    figures_dir = Path(figures_dir)
    y_prob = model.predict([data["X_te_seq"], data["X_te_num"]], verbose=0).flatten()

    if np.isnan(y_prob).any():
        print(f"[ERROR] {name}: {np.isnan(y_prob).sum()} NaN predictions — training diverged.")
        return {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0, "ROC-AUC": 0}

    y_pred = (y_prob >= threshold).astype(int)
    y_true = data["y_te"]

    print(f"\n=== Metrics — {name} (threshold={threshold}) ===")
    print(classification_report(y_true, y_pred, target_names=["Benign(0)", "Attack(1)"]))

    roc_auc = roc_auc_score(y_true, y_prob)
    acc     = float(np.mean(y_pred == y_true))
    cm      = confusion_matrix(y_true, y_pred)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)

    # --- Plot 1: Confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    ax.set_title(f"{name} — Confusion matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    save_fig(fig, figures_dir, "confusion_matrix")

    # --- Plot 2: ROC curve ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"{name} — ROC curve")
    ax.legend(); fig.tight_layout()
    save_fig(fig, figures_dir, "roc_curve")

    # --- Plot 3: Score distribution ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color="steelblue",
            label="Benign", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color="coral",
            label="Attack", density=True)
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
               label=f"Threshold={threshold}")
    ax.set_title(f"{name} — Score distribution")
    ax.set_xlabel("P(attack)"); ax.legend(); fig.tight_layout()
    save_fig(fig, figures_dir, "score_distribution")

    # --- Plot 4: Threshold sweep ---
    ts = np.arange(0.1, 0.91, 0.05)
    recalls, precisions = [], []
    for t in ts:
        yp = (y_prob >= t).astype(int)
        recalls.append(recall_score(y_true, yp, zero_division=0))
        precisions.append(precision_score(y_true, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, recalls,    label="Recall",    color="coral")
    ax.plot(ts, precisions, label="Precision", color="steelblue")
    ax.set_title(f"{name} — Precision & Recall vs Threshold")
    ax.set_xlabel("Threshold"); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    save_fig(fig, figures_dir, "threshold_sweep")

    metrics = {
        "Accuracy":  round(acc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1-Score":  round(f1, 4),
        "ROC-AUC":   round(roc_auc, 4),
    }
    save_json(metrics, config.RESULTS_DIR / f"metrics_{dataset_tag}.json")
    return metrics


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(history, name: str, figures_dir: Path) -> None:
    """Save 3 learning-curve plots (loss, accuracy, precision/recall) as PDF+PNG."""
    figures_dir = Path(figures_dir)
    h      = history.history
    epochs = range(1, len(h["loss"]) + 1)
    best_e = int(np.argmin(h["val_loss"])) + 1

    # Loss
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, h["loss"],     "b-o", ms=3, label="Train Loss")
    ax.plot(epochs, h["val_loss"], "r-o", ms=3, label="Val Loss")
    ax.axvline(best_e, color="gray", ls="--", alpha=0.5, lw=1, label=f"Best (ep {best_e})")
    ax.set_title(f"{name} — Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    save_fig(fig, figures_dir, "loss")

    # Accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, h["accuracy"],     "b-o", ms=3, label="Train Acc")
    ax.plot(epochs, h["val_accuracy"], "r-o", ms=3, label="Val Acc")
    ax.axvline(best_e, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_title(f"{name} — Accuracy"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim([0.5, 1.01]); ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    save_fig(fig, figures_dir, "accuracy")

    # Precision & Recall
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, h["precision"],     "g-o", ms=3, label="Train Precision")
    ax.plot(epochs, h["recall"],        "m-o", ms=3, label="Train Recall")
    ax.plot(epochs, h["val_precision"], "g--", lw=1.5, label="Val Precision")
    ax.plot(epochs, h["val_recall"],    "m--", lw=1.5, label="Val Recall")
    ax.axvline(best_e, color="gray", ls="--", alpha=0.5, lw=1)
    ax.set_title(f"{name} — Precision & Recall")
    ax.set_xlabel("Epoch"); ax.set_ylim([0.5, 1.01])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    save_fig(fig, figures_dir, "precision_recall")

    # Overfit diagnostic
    best_val  = min(h["val_loss"])
    train_val = h["loss"][best_e - 1]
    if not (np.isnan(best_val) or np.isnan(train_val) or train_val == 0):
        gap = (best_val - train_val) / train_val * 100
        tag = ("[WARN] gap > 30% → significant overfitting." if gap > 30
               else "[INFO] gap 15-30% → mild overfitting." if gap > 15
               else "[OK]   gap < 15% → good generalisation.")
        print(f"  Overfit: train_loss={train_val:.4f}, val_loss={best_val:.4f}, gap={gap:+.1f}% {tag}")


# ---------------------------------------------------------------------------
# Per-category breakdown (BCCC)
# ---------------------------------------------------------------------------

def evaluate_per_category(model, data: dict, categories: np.ndarray,
                           name: str, figures_dir: Path,
                           threshold: float = 0.5) -> pd.DataFrame:
    """Per-attack-category detection rates for BCCC.

    Returns a DataFrame with columns: category, support, detection_rate, correct, metric_type.
    """
    from sklearn.model_selection import train_test_split as _tts
    figures_dir = Path(figures_dir)
    y_prob = model.predict([data["X_te_seq"], data["X_te_num"]], verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = data["y_te"]

    # Reconstruct the test indices (same seed as full_pipeline)
    n = len(categories)
    all_idx = np.arange(n)
    _, i_tmp = _tts(all_idx, test_size=(1 - config.TRAIN_RATIO),
                    random_state=42, stratify=data["y_te"] if len(data["y_te"]) == n else None)
    # Simpler: categories was extracted BEFORE split — align by position
    # We rely on the fact that full_pipeline used the same random_state=42
    # and the test set is the last 10 % after the val/test halving.
    # Recompute using the label vector passed in data
    y_all = np.concatenate([data["y_tr"], data["y_val"], data["y_te"]])
    _, i_tmp2 = _tts(np.arange(len(y_all)), test_size=(1 - config.TRAIN_RATIO),
                     random_state=42, stratify=y_all)
    _, i_te = _tts(i_tmp2, test_size=0.5, random_state=42, stratify=y_all[i_tmp2])
    test_categories = categories[i_te] if len(categories) == len(y_all) else categories

    rows = []
    print(f"\n{'category':<22s} {'support':>9s} {'detection_rate':>15s} {'correct':>9s}")
    print("-" * 60)
    for cat in sorted(np.unique(test_categories)):
        mask  = test_categories == cat
        n_cat = int(mask.sum())
        if n_cat == 0:
            continue
        cat_true = y_true[mask]
        cat_pred = y_pred[mask]
        if str(cat).lower() == "benign":
            correct = int((cat_pred == 0).sum())
            mtype   = "specificity"
        else:
            correct = int((cat_pred == 1).sum())
            mtype   = "recall"
        rate = correct / n_cat
        rows.append({"category": cat, "support": n_cat,
                     "detection_rate": round(rate, 4),
                     "correct": correct, "metric_type": mtype})
        print(f"{str(cat):<22s} {n_cat:>9,} {rate:>14.4f} ({mtype}) {correct:>9,}")

    df = pd.DataFrame(rows)
    save_json(df.to_dict("records"), config.RESULTS_DIR / f"per_category_{name}.json")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#4C72B0" if r["metric_type"] == "specificity" else "#DD8452" for r in rows]
    ax.barh(df["category"], df["detection_rate"], color=colors)
    ax.axvline(0.9, color="red", ls="--", lw=1, label="90% reference")
    ax.set_xlabel("Detection rate")
    ax.set_title(f"{name} — Per-category detection rate")
    ax.set_xlim([0, 1.05]); ax.legend(); fig.tight_layout()
    save_fig(fig, figures_dir, "per_category_detection")
    return df


# ---------------------------------------------------------------------------
# Ablation comparison
# ---------------------------------------------------------------------------

def ablation_comparison(results: Dict[str, dict],
                        dataset_tag: str,
                        figures_dir: Path) -> pd.DataFrame:
    """Build and save an ablation comparison table + bar chart.

    Args:
        results: {variant_name: metrics_dict}  — e.g.
                 {"FFN-Only": {...}, "Encoder-Only": {...}, "Full (paper)": {...}}
        dataset_tag: used for file naming.
        figures_dir: where to save the PDF plot.
    """
    figures_dir = Path(figures_dir)
    df = pd.DataFrame(results).T
    df.index.name = "Variant"
    print(f"\n=== Ablation study — {dataset_tag} ===")
    print(df.to_string())

    # Bar chart
    metrics_to_plot = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
                       if c in df.columns]
    x = np.arange(len(df))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, m in enumerate(metrics_to_plot):
        ax.bar(x + i * width, df[m].astype(float), width, label=m, color=colors[i])
    ax.set_xticks(x + width * (len(metrics_to_plot) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=10, ha="right")
    ax.set_ylim([0.5, 1.05])
    ax.set_title(f"Ablation study — {dataset_tag}")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, figures_dir, f"ablation_{dataset_tag}")

    save_json(results, config.RESULTS_DIR / f"ablation_{dataset_tag}.json")
    return df


# ---------------------------------------------------------------------------
# Statistical significance: McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                 name_a: str = "Model A", name_b: str = "Model B") -> dict:
    """McNemar's test to compare two classifiers on the same test set.

    Computes the 2×2 contingency table of disagreements between A and B and
    applies the exact McNemar test (with continuity correction for small n).

    Returns:
        dict with keys: chi2, p_value, n_ab (A correct, B wrong),
        n_ba (A wrong, B correct), significant (bool at α=0.05).
    """
    # n_ab: A correct, B wrong  |  n_ba: A wrong, B correct
    n_ab = int(np.sum((y_pred_a == y_true) & (y_pred_b != y_true)))
    n_ba = int(np.sum((y_pred_a != y_true) & (y_pred_b == y_true)))
    n    = n_ab + n_ba

    if n == 0:
        print(f"McNemar ({name_a} vs {name_b}): identical predictions — no test needed.")
        return {"chi2": 0.0, "p_value": 1.0, "n_ab": 0, "n_ba": 0, "significant": False}

    # Continuity-corrected chi² statistic
    stat  = (abs(n_ab - n_ba) - 1) ** 2 / (n_ab + n_ba)
    p_val = float(1 - chi2.cdf(stat, df=1))
    sig   = p_val < 0.05

    print(f"\nMcNemar test: {name_a} vs {name_b}")
    print(f"  n_ab (A✓, B✗) = {n_ab}  |  n_ba (A✗, B✓) = {n_ba}")
    print(f"  χ² = {stat:.4f}  |  p-value = {p_val:.4f}  |  "
          f"{'Significant (α=0.05)' if sig else 'Not significant (α=0.05)'}")

    return {"chi2": round(stat, 4), "p_value": round(p_val, 6),
            "n_ab": n_ab, "n_ba": n_ba, "significant": sig}


# ---------------------------------------------------------------------------
# Final summary bar chart (all datasets)
# ---------------------------------------------------------------------------

def final_summary_plot(summary: dict, figures_dir: Path) -> None:
    """Bar chart comparing all datasets × metrics."""
    figures_dir = Path(figures_dir)
    df = pd.DataFrame(summary).T
    x = np.arange(len(df))
    width = 0.18
    metrics_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in df.columns]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metrics_cols):
        ax.bar(x + i * width, df[m].astype(float), width, label=m, color=colors[i])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df.index, fontsize=9, rotation=10, ha="right")
    ax.set_ylim([0.5, 1.05])
    ax.set_title("Final metric comparison across datasets")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, figures_dir, "00_final_comparison")
