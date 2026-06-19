"""Exploratory Data Analysis — one PDF+PNG per plot.

Exposes:
    eda_marques(df, figures_dir)
    eda_bccc(df, figures_dir)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config
from .utils import save_fig

TARGET_COL = config.TARGET_COL


# ---------------------------------------------------------------------------
# Marques EDA
# ---------------------------------------------------------------------------

def eda_marques(df: pd.DataFrame, figures_dir: Path) -> None:
    """6 EDA plots for Marques et al. dataset."""
    figures_dir = Path(figures_dir)
    print("=== EDA — Marques et al. ===")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=["#2196F3", "#F44336"])
    ax.set_title("Marques — Class distribution")
    ax.set_xlabel("Class (0=Benign, 1=Attack)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0 (Benign)", "1 (Attack)"], rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); save_fig(fig, figures_dir, "01_class_distribution")

    # 2. DNS record type
    if "DNSRecordType" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        df["DNSRecordType"].value_counts().head(10).plot(kind="bar", ax=ax, color="#4CAF50")
        ax.set_title("Marques — DNS Record Type"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); save_fig(fig, figures_dir, "02_dns_record_type")

    # 3. Top TLD
    if "TLD" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["TLD"].value_counts().head(15).plot(kind="bar", ax=ax, color="#FF9800")
        ax.set_title("Marques — Top 15 TLD"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); save_fig(fig, figures_dir, "03_top_tld")

    # 4. Domain length
    col = "DomainLength" if "DomainLength" in df.columns else None
    if col is None:
        num_c = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Class"]
        if num_c:
            col = num_c[0]
    if col:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df[col].dropna(), bins=40, color="#9C27B0", edgecolor="white", alpha=0.7)
        ax.set_title(f"Marques — {col} distribution"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); save_fig(fig, figures_dir, "04_domain_length")

    # 5. Entropy distribution
    if "Entropy" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df["Entropy"].dropna(), bins=30, color="#00BCD4", edgecolor="white", alpha=0.7)
        ax.set_title("Marques — Entropy distribution"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); save_fig(fig, figures_dir, "05_entropy")

    # 6. Country code
    if "CountryCode" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        df["CountryCode"].replace("null", np.nan).dropna().value_counts().head(10).plot(
            kind="bar", ax=ax, color="#607D8B")
        ax.set_title("Marques — Top 10 CountryCode"); ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); save_fig(fig, figures_dir, "06_country_code")

    print("  EDA Marques done.")


# ---------------------------------------------------------------------------
# BCCC EDA
# ---------------------------------------------------------------------------

def eda_bccc(df: pd.DataFrame, figures_dir: Path) -> None:
    """2 EDA plots for BCCC-CIC-Bell-DNS-2024."""
    figures_dir = Path(figures_dir)
    print("=== EDA — BCCC-CIC-Bell-DNS-2024 ===")

    # 1. Attack category distribution (16 sub-types)
    if "attack_category" in df.columns:
        counts = df["attack_category"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(counts.index.astype(str), counts.values, color="#C44E52")
        for i, v in enumerate(counts.values):
            ax.text(v, i, f"  {v:,}", va="center", fontsize=8)
        ax.set_title("BCCC — Attack category distribution")
        ax.set_xlabel("Count"); ax.invert_yaxis()
        fig.tight_layout(); save_fig(fig, figures_dir, "01_category_distribution")

    # 2. Binary label distribution
    label_col = "label" if "label" in df.columns else TARGET_COL
    if label_col in df.columns:
        counts = df[label_col].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"])
        for i, v in enumerate(counts.values):
            ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center", fontweight="bold")
        ax.set_title("BCCC — Binary class distribution")
        ax.set_xlabel("Class (0=Benign, 1=Malicious)"); ax.set_ylabel("Count")
        fig.tight_layout(); save_fig(fig, figures_dir, "02_binary_class_distribution")

    print("  EDA BCCC done.")
