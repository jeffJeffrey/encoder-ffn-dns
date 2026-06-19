"""Preprocessing pipeline for Marques and BCCC datasets.

Exposes:
    preprocess_marques(df)    -> (X_num, X_text, y, num_col_names, text_cols)
    preprocess_bccc(df)       -> (X_num, X_text, y, num_col_names, text_cols, categories)
    full_pipeline(...)        -> data dict ready for training
"""
from __future__ import annotations

import ast
import math
import re
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from . import config

TARGET_COL = config.TARGET_COL
NOT_A_DNS  = "not a dns flow"

# ===========================================================================
# General helpers
# ===========================================================================

def report_nulls(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    report = pd.DataFrame({"Nulls": null_counts, "%": null_pct})
    report = report[report["Nulls"] > 0].sort_values("Nulls", ascending=False)
    if not report.empty:
        print(f"  [WARNING] {name} — missing values:\n{report.to_string()}")
    else:
        print(f"  No missing values in {name}")
    return report


def fix_null_strings(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].replace(
            ["null", "NULL", "None", "none", "nan", "NaN", "NAN", "na", "NA", ""],
            np.nan
        )
    return df


def convert_bool_strings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        df[col] = df[col].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
        )
    return df


def fix_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[num_cols]).sum().sum()
    if n_inf > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET_COL]
    all_nan = [c for c in num_cols if df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        num_cols = [c for c in num_cols if c not in all_nan]
    n_nan = df[num_cols].isna().sum().sum()
    if n_nan > 0:
        for col in num_cols:
            if df[col].isna().any():
                med = df[col].median()
                df[col] = df[col].fillna(0.0 if pd.isna(med) else med)
    return df


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    str_cols = df.select_dtypes(include="object").columns
    n_nan = df[str_cols].isna().sum().sum()
    if n_nan > 0:
        df[str_cols] = df[str_cols].fillna("UNK")
    return df


# ===========================================================================
# CIC / BCCC set/list parsers
# ===========================================================================

def parse_set_string(val):
    if isinstance(val, (set, frozenset)):
        return set(val)
    if isinstance(val, list):
        return set(val)
    s = str(val).strip()
    if s in ("", "nan", "NaN", "None", "none", "set()", "NAN"):
        return set()
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (set, frozenset)):
            return set(result)
        if isinstance(result, list):
            return set(result)
        if isinstance(result, dict):
            return set(result.keys())
        return {result}
    except (ValueError, SyntaxError):
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"|\b(\d+\.?\d*)\b", s)
        extracted = set()
        for t in tokens:
            v = t[0] or t[1] or t[2]
            if v:
                extracted.add(v.strip())
        return extracted


def parse_list_string(val):
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, (set, frozenset)):
        return list(val)
    if pd.isna(val):
        return []
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null", NOT_A_DNS, "set()", "not a tcp connection"):
        return []
    try:
        result = ast.literal_eval(s)
        if isinstance(result, (list, tuple, set, frozenset)):
            return list(result)
        if isinstance(result, dict):
            return list(result.keys())
        return [result]
    except (ValueError, SyntaxError):
        tokens = re.findall(r"'([^']+)'|\"([^\"]+)\"|\b(\d+\.?\d*)\b", s)
        return [t[0] or t[1] or t[2] for t in tokens if (t[0] or t[1] or t[2])]


def parse_dict_string(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return {}
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null", NOT_A_DNS):
        return {}
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def rr_type_to_tokens(val) -> str:
    s = parse_set_string(val)
    if not s:
        return "UNK_RR"
    tokens = sorted([str(t).strip().upper().replace(" ", "").replace("'", "") for t in s if str(t).strip()])
    return " ".join(tokens) if tokens else "UNK_RR"


def country_to_tokens(val) -> str:
    s = parse_set_string(val)
    if not s:
        return "UNK_CC"
    tokens = sorted([str(t).strip().upper() for t in s
                     if str(t).strip().isalpha() and 1 <= len(str(t).strip()) <= 3])
    return " ".join(tokens[:5]) if tokens else "UNK_CC"


def tld_clean(val) -> str:
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none", NOT_A_DNS):
        return "UNK_TLD"
    return str(val).strip().lower().lstrip(".").upper() or "UNK_TLD"


def domain_to_suffix(val) -> str:
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none", NOT_A_DNS):
        return "UNK_DOM"
    parts = str(val).strip().lower().rstrip(".").split(".")
    return ".".join(parts[-2:]).upper() if len(parts) >= 2 else (parts[0].upper() or "UNK_DOM")


def sld_clean(val) -> str:
    if pd.isna(val) or str(val).strip().lower() in ("", "nan", "none", NOT_A_DNS):
        return "UNK_SLD"
    parts = str(val).strip().lower().lstrip(".").split(".")
    s = ".".join(parts[-2:]) if len(parts) >= 2 else parts[0]
    return s.upper() if s else "UNK_SLD"


def char_distribution_entropy(val) -> float:
    d = parse_dict_string(val)
    if not d:
        return 0.0
    total = sum(d.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for v in d.values():
        p = v / total
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


def list_count(val) -> int:
    return len(parse_list_string(val))


_RR_CODE = {
    1:"A",2:"NS",5:"CNAME",6:"SOA",12:"PTR",15:"MX",16:"TXT",28:"AAAA",
    33:"SRV",35:"NAPTR",41:"OPT",43:"DS",46:"RRSIG",47:"NSEC",48:"DNSKEY",
    50:"NSEC3",51:"NSEC3PARAM",65:"HTTPS",255:"ANY"
}

def rr_type_codes_to_tokens(val) -> str:
    items = parse_list_string(val)
    tokens = []
    for x in items:
        try:
            tokens.append(_RR_CODE.get(int(float(x)), f"RR{int(float(x))}"))
        except (ValueError, TypeError):
            s = str(x).strip().upper()
            if s:
                tokens.append(s)
    return " ".join(sorted(set(tokens))) if tokens else "UNK_RR"


def rr_class_codes_to_tokens(val) -> str:
    code_to_name = {1:"IN",2:"CS",3:"CH",4:"HS",254:"NONE",255:"ANY"}
    items = parse_list_string(val)
    tokens = []
    for x in items:
        try:
            tokens.append(code_to_name.get(int(float(x)), f"CL{int(float(x))}"))
        except (ValueError, TypeError):
            s = str(x).strip().upper()
            if s:
                tokens.append(s)
    return " ".join(sorted(set(tokens))) if tokens else "UNK_CL"


def domain_ngram_tokens(val, top_k: int = 5) -> str:
    items = parse_list_string(val)
    tokens = [str(t).strip().lower().replace(" ", "_")
               for t in items[:top_k] if str(t).strip() and not str(t).strip().startswith(".")]
    return " ".join(tokens) if tokens else "UNK_NG"


# ===========================================================================
# Marques preprocessing
# ===========================================================================

def preprocess_marques(df: pd.DataFrame):
    """Full preprocessing pipeline for Marques et al. dataset.

    Returns (X_num, X_text, y, num_col_names, text_cols).
    """
    df = df.copy()
    print("=== Preprocessing Marques ===")

    # 1. Deduplication
    before = len(df)
    df = df.groupby(list(df.columns), sort=False).head(config.DEDUP_MAX_OCCURRENCES)
    df = df.reset_index(drop=True)
    print(f"  [1/6] Dedup: {before-len(df)} rows removed, {len(df)} kept")

    # 2. Null strings
    df = fix_null_strings(df)

    # 3. Bool strings
    bool_cols = []
    for col in df.select_dtypes(include="object").columns:
        if col == "Class":
            continue
        uniq = set(df[col].dropna().astype(str).str.lower().unique())
        if uniq.issubset({"true", "false", "0", "1", "yes", "no"}):
            bool_cols.append(col)
    if bool_cols:
        df = convert_bool_strings(df, bool_cols)

    # 4. Infinities
    df = fix_inf_values(df)

    # 5. Text columns
    str_cols = df.select_dtypes(include="object").columns.tolist()
    text_cols = []
    for col in str_cols:
        if col in ("Class", TARGET_COL):
            continue
        null_pct = df[col].isna().mean()
        n_unique = df[col].nunique()
        if null_pct < 0.50 and 2 <= n_unique <= 500:
            text_cols.append(col)
    if "DNSRecordType" in df.columns and "DNSRecordType" not in text_cols:
        text_cols.insert(0, "DNSRecordType")
    print(f"  [5/6] Text cols: {text_cols}")

    # 6. Imputation
    df = impute_numeric(df)
    df = impute_categorical(df)

    # Label
    if "Class" in df.columns:
        df = df.rename(columns={"Class": TARGET_COL})
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X_num, X_text, y, num_col_names = _split_features(df, text_cols, TARGET_COL)
    print(f"  X_num: {X_num.shape}  |  X_text: {len(X_text)}  |  y: {y.shape}")
    print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X_num, X_text, y, num_col_names, text_cols


# ===========================================================================
# BCCC preprocessing
# ===========================================================================

def preprocess_bccc(df: pd.DataFrame):
    """Full preprocessing pipeline for BCCC-CIC-Bell-DNS-2024.

    Returns (X_num, X_text, y, num_col_names, text_cols, categories).
    """
    df = df.copy()
    print("=== Preprocessing BCCC ===")

    # 1. Filter non-DNS rows
    dns_marker_cols = [c for c in ["domain_name", "top_level_domain", "second_level_domain"]
                       if c in df.columns]
    if dns_marker_cols:
        before = len(df)
        mask = pd.Series(False, index=df.index)
        for col in dns_marker_cols:
            mask |= df[col].astype(str).str.strip().str.lower() == NOT_A_DNS
        df = df[~mask].reset_index(drop=True)
        print(f"  [1] Non-DNS rows removed: {before-len(df):,}")

    # 2. Null strings
    df = fix_null_strings(df)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace({NOT_A_DNS: np.nan, "not a tcp connection": np.nan})

    # 3. Parse list/dict columns
    if "query_resource_record_type" in df.columns:
        df["query_rr_type_text"]  = df["query_resource_record_type"].apply(rr_type_codes_to_tokens)
        df["query_rr_type_count"] = df["query_resource_record_type"].apply(list_count)
        df = df.drop(columns=["query_resource_record_type"])
    if "ans_resource_record_type" in df.columns:
        df["ans_rr_type_text"]  = df["ans_resource_record_type"].apply(rr_type_codes_to_tokens)
        df["ans_rr_type_count"] = df["ans_resource_record_type"].apply(list_count)
        df = df.drop(columns=["ans_resource_record_type"])
    if "query_resource_record_class" in df.columns:
        df["query_rr_class_text"] = df["query_resource_record_class"].apply(rr_class_codes_to_tokens)
        df = df.drop(columns=["query_resource_record_class"])
    if "ans_resource_record_class" in df.columns:
        df["ans_rr_class_text"] = df["ans_resource_record_class"].apply(rr_class_codes_to_tokens)
        df = df.drop(columns=["ans_resource_record_class"])
    if "bi_gram_domain_name" in df.columns:
        df["bi_gram_tokens"] = df["bi_gram_domain_name"].apply(lambda v: domain_ngram_tokens(v, 5))
        df = df.drop(columns=["bi_gram_domain_name"])
    for c in ["uni_gram_domain_name", "tri_gram_domain_name"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    if "character_distribution" in df.columns:
        df["char_dist_entropy"] = df["character_distribution"].apply(char_distribution_entropy)
        df["char_dist_size"]    = df["character_distribution"].apply(list_count)
        df = df.drop(columns=["character_distribution"])

    # 4. Clean domain / TLD / SLD
    if "top_level_domain" in df.columns:
        df["tld_clean"] = df["top_level_domain"].apply(tld_clean)
        df = df.drop(columns=["top_level_domain"])
    if "second_level_domain" in df.columns:
        df["sld_clean"] = df["second_level_domain"].apply(sld_clean)
        df = df.drop(columns=["second_level_domain"])
    if "domain_name" in df.columns:
        df["domain_suffix"] = df["domain_name"].apply(domain_to_suffix)
        df = df.drop(columns=["domain_name"])

    # 5. Coerce residual objects to numeric
    KEEP_AS_TEXT = {"tld_clean","sld_clean","domain_suffix",
                    "query_rr_type_text","ans_rr_type_text",
                    "query_rr_class_text","ans_rr_class_text",
                    "bi_gram_tokens","attack_category"}
    for col in list(df.select_dtypes(include="object").columns):
        if col in KEEP_AS_TEXT or col == "label":
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        notna_orig = max(df[col].notna().sum(), 1)
        if coerced.notna().sum() / notna_orig > 0.8:
            df[col] = coerced
        else:
            df = df.drop(columns=[col])

    # 6. Drop quasi-constant
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "label"]
    variances = df[num_cols].replace([np.inf, -np.inf], np.nan).var()
    low_var = variances[variances < 1e-10].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)

    # 7. Infinities
    df = fix_inf_values(df)

    # 8. Dedup
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  [8] Dedup: {before-len(df):,} rows removed")

    # 9. log1p on heavy-tailed numerics
    num_cols_log = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("label", TARGET_COL)]
    for col in num_cols_log:
        s = df[col].dropna()
        if len(s) > 0 and s.min() >= 0 and s.skew() > 3 and (s.max() - s.min()) > 100:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 10. Drop highly correlated (|corr| > 0.95)
    num_cols_corr = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ("label", TARGET_COL)]
    if len(num_cols_corr) > 1:
        sample   = df[num_cols_corr].sample(n=min(50_000, len(df)), random_state=42)
        corr_mat = sample.corr().abs()
        upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
        to_drop  = [c for c in upper.columns if any(upper[c] > 0.95)]
        if to_drop:
            df = df.drop(columns=to_drop)

    # 11. Imputation
    df = impute_numeric(df)
    df = impute_categorical(df)

    # Safety: no NaN/inf
    num_final = df.select_dtypes(include=[np.number]).columns
    df[num_final] = df[num_final].fillna(0.0)

    # Extract category sidecar BEFORE label rename
    categories = None
    if "attack_category" in df.columns:
        categories = df["attack_category"].values.copy()
        df = df.drop(columns=["attack_category"])

    # Label
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df = df.rename(columns={"label": TARGET_COL})

    # Text columns
    text_cols = [c for c in ["query_rr_type_text","ans_rr_type_text",
                              "query_rr_class_text","ans_rr_class_text",
                              "tld_clean","sld_clean","domain_suffix","bi_gram_tokens"]
                 if c in df.columns]

    X_num, X_text, y, num_col_names = _split_features(df, text_cols, TARGET_COL)
    print(f"  X_num: {X_num.shape}  |  X_text: {len(X_text)}  |  y: {y.shape}")
    print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X_num, X_text, y, num_col_names, text_cols, categories


# ===========================================================================
# Feature split helpers
# ===========================================================================

def _build_text_sequence(df: pd.DataFrame, text_cols: List[str]) -> pd.Series:
    seq = None
    for col in text_cols:
        if col not in df.columns:
            continue
        part = df[col].fillna("UNK").astype(str).str.strip().str.upper()
        seq  = part if seq is None else seq + " " + part
    if seq is None:
        seq = pd.Series(["UNK"] * len(df), index=df.index)
    return seq


def _split_features(df: pd.DataFrame, text_cols: List[str], target_col: str):
    y      = df[target_col].values.astype(int)
    X_text = _build_text_sequence(df, text_cols)
    drop   = [c for c in text_cols if c in df.columns] + [target_col]
    X_num  = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    names  = list(X_num.columns)
    return X_num.values.astype(np.float32), X_text, y, names


# ===========================================================================
# Train / Val / Test pipeline
# ===========================================================================

def full_pipeline(X_num: np.ndarray, X_text: pd.Series, y: np.ndarray,
                  dataset_name: str = "Dataset",
                  max_vocab: int = config.MAX_VOCAB_MARQUES,
                  max_seq: int = config.MAX_SEQ_LEN) -> dict:
    """Stratified 80/10/10 split + MinMaxScaler + Keras Tokenizer.

    Returns a dict with all arrays and fitted transformers.

    Split note: the paper states an 80/20 train/validation split for the
    offline training phase. In this implementation we use 80/10/10
    (train/val/test) so that a held-out test set is available for unbiased
    evaluation. The validation set (10%) is used for early-stopping and
    hyper-parameter selection; the test set (10%) is used solely for the
    final metrics reported in the paper. The resulting test-set sizes
    (~9 000 for Marques, ~97 000 for BCCC) are therefore smaller than a
    naïve 20% split but remain statistically sufficient for the reported
    classification metrics.
    """
    print(f"\n=== Pipeline {dataset_name} ===")

    # 1. Stratified 80 / 10 / 10 split
    X_tr_n, X_tmp_n, X_tr_t, X_tmp_t, y_tr, y_tmp = train_test_split(
        X_num, X_text, y,
        test_size=(1 - config.TRAIN_RATIO), random_state=42, stratify=y
    )
    X_val_n, X_te_n, X_val_t, X_te_t, y_val, y_te = train_test_split(
        X_tmp_n, X_tmp_t, y_tmp,
        test_size=0.5, random_state=42, stratify=y_tmp
    )
    print(f"  Split: {len(y_tr):,} train | {len(y_val):,} val | {len(y_te):,} test")

    # 2. MinMaxScaler — fit on TRAIN only
    for arr in (X_tr_n, X_val_n, X_te_n):
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    scaler = MinMaxScaler()
    X_tr_n  = scaler.fit_transform(X_tr_n).astype(np.float32)
    X_val_n = scaler.transform(X_val_n).astype(np.float32)
    X_te_n  = scaler.transform(X_te_n).astype(np.float32)
    for arr in (X_tr_n, X_val_n, X_te_n):
        np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0, copy=False)

    # 3. Tokenizer — fit on TRAIN only
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr_t.tolist())
    vocab_size = min(len(tokenizer.word_index) + 1, max_vocab)
    print(f"  Vocab: {len(tokenizer.word_index)} tokens → using {vocab_size}")

    def _tok(texts):
        seqs = tokenizer.texts_to_sequences(texts.tolist())
        return pad_sequences(seqs, maxlen=max_seq, padding="post",
                             truncating="post").astype(np.int32)

    return {
        "X_tr_num":  X_tr_n,  "X_val_num":  X_val_n,  "X_te_num":  X_te_n,
        "X_tr_seq":  _tok(X_tr_t), "X_val_seq": _tok(X_val_t), "X_te_seq": _tok(X_te_t),
        "y_tr": y_tr, "y_val": y_val, "y_te": y_te,
        "scaler": scaler, "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "n_num_features": X_tr_n.shape[1],
        "max_seq": max_seq,
    }
