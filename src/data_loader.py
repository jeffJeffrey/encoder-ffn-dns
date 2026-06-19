"""Dataset downloading with automatic caching.

- Marques et al.: downloaded from GitHub via HTTPS.
- BCCC-CIC-Bell-DNS-2024: downloaded via kagglehub.
"""
from __future__ import annotations

import gc
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Leakage columns to drop from BCCC (WHOIS / meta fields)
# ---------------------------------------------------------------------------
_LEAKAGE_LC = {
    'flow_id', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
    'whoisdomainname', 'domainemail', 'domainregistrar',
    'domaincreationdate', 'domainexpirationdate', 'domainage',
    'domaincountry', 'domaindnssec', 'domainorganization',
    'domainaddress', 'domaincity', 'domainstate', 'domainzipcode',
    'domainnameservers', 'domainupdateddate',
    'dns_whois_domain_name', 'dns_domain_email', 'dns_domain_registrar',
    'dns_domain_creation_date', 'dns_domain_expiration_date',
    'dns_domain_age', 'dns_domain_country', 'dns_domain_dnssec',
    'dns_domain_organization', 'dns_domain_address',
    'dns_domain_city', 'dns_domain_state', 'dns_domain_zipcode',
    'dns_domain_name_servers', 'dns_domain_updated_date',
}


# ---------------------------------------------------------------------------
# Marques et al.
# ---------------------------------------------------------------------------

def get_marques() -> pd.DataFrame:
    """Load the Marques et al. DNS dataset from the local datasets/ folder."""
    path = config.MARQUES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Marques dataset not found at: {path}\n"
            f"Make sure 'marques_dns_dataset.csv' is inside the 'datasets/' folder "
            f"at the root of the repository and has been committed to git."
        )

    print(f"Loading Marques dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")

    # Détecter la colonne label sans supposer son nom exact
    label_col = next(
        (c for c in df.columns if c.strip().lower() in ('class', 'label', 'target')),
        None
    )
    if label_col:
        print(f"  Label: '{label_col}' → {df[label_col].value_counts().to_dict()}")
    else:
        print(f"  [WARNING] Aucune colonne label trouvée: {list(df.columns)}")

    return df   # ← retourne le DataFrame brut, 'Class' reste 'Class'
                #   le rename vers TARGET_COL se fait dans preprocessing.py


# ---------------------------------------------------------------------------
# BCCC-CIC-Bell-DNS-2024
# ---------------------------------------------------------------------------

def _count_rows(path: str) -> int:
    try:
        n = int(subprocess.check_output(["wc", "-l", path]).split()[0]) - 1
        return max(n, 0)
    except Exception:
        with open(path, "rb") as f:
            return sum(1 for _ in f) - 1


def _file_category(path: str):
    name   = os.path.basename(path).lower()
    parent = os.path.basename(os.path.dirname(path)).lower()
    if "benign" in name:
        return "benign", 0
    if "exf" in parent or any(k in name for k in ["heavy_", "light_"]):
        return "exfiltration", 1
    for kw in ("malware", "phishing", "spam"):
        if kw in name:
            return kw, 1
    return "other_malicious", 1


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def get_bccc(force: bool = False) -> pd.DataFrame:
    """Download (or load from cache) the BCCC-CIC-Bell-DNS-2024 dataset."""
    bccc_dir = config.BCCC_DIR
    csv_files = sorted(glob.glob(str(bccc_dir / "**" / "*.csv"), recursive=True))

    if not csv_files or force:
        print("Downloading BCCC-CIC-Bell-DNS-2024 via kagglehub…")
        import kagglehub, shutil
        path = kagglehub.dataset_download(config.BCCC_KAGGLE_SLUG)
        if bccc_dir.exists():
            shutil.rmtree(bccc_dir)
        shutil.copytree(path, bccc_dir)
        csv_files = sorted(glob.glob(str(bccc_dir / "**" / "*.csv"), recursive=True))
        print(f"  Downloaded {len(csv_files)} CSV files → {bccc_dir}")
    else:
        print(f"BCCC cache found: {len(csv_files)} CSV files in {bccc_dir}")

    # --- Preflight ---
    first_cols  = pd.read_csv(csv_files[0], nrows=0).columns.tolist()
    keep_cols   = [c for c in first_cols if c.lower() not in _LEAKAGE_LC]
    file_info   = []
    total_benign = 0
    for p in csv_files:
        cat, lbl = _file_category(p)
        n = _count_rows(p)
        file_info.append({"path": p, "category": cat, "label": lbl, "n_rows": n})
        if lbl == 0:
            total_benign += n

    target_benign = min(config.BCCC_MAX_BENIGN_ROWS, total_benign)
    benign_rate   = target_benign / total_benign if total_benign > 0 else 1.0
    rng = np.random.default_rng(42)

    dfs = []
    print(f"Stream loading {len(file_info)} files (benign rate={benign_rate:.3f})…")
    for fi in file_info:
        rate       = benign_rate if fi["label"] == 0 else 1.0
        actual_cols = pd.read_csv(fi["path"], nrows=0).columns.tolist()
        uc         = [c for c in keep_cols if c in actual_cols]
        parts, kept = [], 0
        reader = pd.read_csv(fi["path"], usecols=uc, low_memory=False,
                             chunksize=config.BCCC_CHUNK_SIZE, on_bad_lines="skip")
        for chunk in reader:
            chunk = _downcast(chunk)
            if rate < 1.0:
                mask  = rng.random(len(chunk)) < rate
                chunk = chunk[mask]
            if len(chunk) > 0:
                parts.append(chunk)
                kept += len(chunk)
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()

        # Assign label / category
        for lbl_col in ("label", "Label", "LABEL", "class", "Class"):
            if lbl_col in df.columns:
                s = df[lbl_col].astype(str).str.lower().str.strip()
                df["attack_category"] = s.where(s != "", fi["category"])
                df["label"] = (~s.isin(["benign", "0", "normal", "legitimate"])).astype(int)
                if lbl_col != "label":
                    df = df.drop(columns=[lbl_col])
                break
        else:
            df["label"]           = fi["label"]
            df["attack_category"] = fi["category"]

        dfs.append(df)

    df_bccc = pd.concat(dfs, ignore_index=True, sort=False)
    df_bccc.columns = [c.strip() for c in df_bccc.columns]
    del dfs
    gc.collect()

    print(f"BCCC loaded — shape: {df_bccc.shape}")
    print(f"  Label dist:    {df_bccc['label'].value_counts().to_dict()}")
    print(f"  Categories:    {df_bccc['attack_category'].value_counts().to_dict()}")
    return df_bccc
