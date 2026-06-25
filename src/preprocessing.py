"""Preprocessing pipeline for Marques and BCCC datasets."""
from __future__ import annotations

import ast
import math
import re
import warnings
from typing import List

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

def report_nulls(df, name="Dataset"):
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    report = pd.DataFrame({"Nulls": null_counts, "%": null_pct})
    report = report[report["Nulls"] > 0].sort_values("Nulls", ascending=False)
    if not report.empty:
        print(f"  [WARNING] {name} — missing values:\n{report.to_string()}")
    else:
        print(f"  No missing values in {name}")
    return report

def fix_null_strings(df):
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].replace(
            ["null","NULL","None","none","nan","NaN","NAN","na","NA",""], np.nan)
    return df

def convert_bool_strings(df, cols):
    for col in cols:
        df[col] = df[col].astype(str).str.lower().map(
            {"true":1,"false":0,"1":1,"0":0,"yes":1,"no":0})
    return df

def fix_inf_values(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if np.isinf(df[num_cols]).sum().sum() > 0:
        df[num_cols] = df[num_cols].replace([np.inf,-np.inf], np.nan)
    return df

def impute_numeric(df):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET_COL]
    all_nan = [c for c in num_cols if df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        num_cols = [c for c in num_cols if c not in all_nan]
    for col in num_cols:
        if df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(med) else med)
    return df

def impute_categorical(df):
    str_cols = df.select_dtypes(include="object").columns
    if df[str_cols].isna().sum().sum() > 0:
        df[str_cols] = df[str_cols].fillna("UNK")
    return df


# ===========================================================================
# BCCC parsers (unchanged)
# ===========================================================================

def parse_set_string(val):
    if isinstance(val,(set,frozenset)): return set(val)
    if isinstance(val,list): return set(val)
    s=str(val).strip()
    if s in ("","nan","NaN","None","none","set()","NAN"): return set()
    try:
        r=ast.literal_eval(s)
        if isinstance(r,(set,frozenset)): return set(r)
        if isinstance(r,list): return set(r)
        if isinstance(r,dict): return set(r.keys())
        return {r}
    except: pass
    tokens=re.findall(r"'([^']+)'|\"([^\"]+)\"|\b(\d+\.?\d*)\b",s)
    return {(t[0] or t[1] or t[2]).strip() for t in tokens if (t[0] or t[1] or t[2])}

def parse_list_string(val):
    if isinstance(val,(list,tuple)): return list(val)
    if isinstance(val,(set,frozenset)): return list(val)
    if pd.isna(val): return []
    s=str(val).strip()
    if s.lower() in ("","nan","none","null",NOT_A_DNS,"set()","not a tcp connection"): return []
    try:
        r=ast.literal_eval(s)
        if isinstance(r,(list,tuple,set,frozenset)): return list(r)
        if isinstance(r,dict): return list(r.keys())
        return [r]
    except: pass
    tokens=re.findall(r"'([^']+)'|\"([^\"]+)\"|\b(\d+\.?\d*)\b",s)
    return [t[0] or t[1] or t[2] for t in tokens if (t[0] or t[1] or t[2])]

def parse_dict_string(val):
    if isinstance(val,dict): return val
    if pd.isna(val): return {}
    s=str(val).strip()
    if s.lower() in ("","nan","none","null",NOT_A_DNS): return {}
    try: r=ast.literal_eval(s); return r if isinstance(r,dict) else {}
    except: return {}

def rr_type_codes_to_tokens(val):
    _RR={1:"A",2:"NS",5:"CNAME",6:"SOA",12:"PTR",15:"MX",16:"TXT",28:"AAAA",
         33:"SRV",35:"NAPTR",41:"OPT",43:"DS",46:"RRSIG",47:"NSEC",48:"DNSKEY",
         50:"NSEC3",51:"NSEC3PARAM",65:"HTTPS",255:"ANY"}
    items=parse_list_string(val); tokens=[]
    for x in items:
        try: tokens.append(_RR.get(int(float(x)),f"RR{int(float(x))}"))
        except: s=str(x).strip().upper(); tokens.append(s) if s else None
    return " ".join(sorted(set(tokens))) if tokens else "UNK_RR"

def rr_class_codes_to_tokens(val):
    _CL={1:"IN",2:"CS",3:"CH",4:"HS",254:"NONE",255:"ANY"}
    items=parse_list_string(val); tokens=[]
    for x in items:
        try: tokens.append(_CL.get(int(float(x)),f"CL{int(float(x))}"))
        except: s=str(x).strip().upper(); tokens.append(s) if s else None
    return " ".join(sorted(set(tokens))) if tokens else "UNK_CL"

def tld_clean(val):
    if pd.isna(val) or str(val).strip().lower() in ("","nan","none",NOT_A_DNS): return "UNK_TLD"
    return str(val).strip().lower().lstrip(".").upper() or "UNK_TLD"

def domain_to_suffix(val):
    if pd.isna(val) or str(val).strip().lower() in ("","nan","none",NOT_A_DNS): return "UNK_DOM"
    parts=str(val).strip().lower().rstrip(".").split(".")
    return ".".join(parts[-2:]).upper() if len(parts)>=2 else (parts[0].upper() or "UNK_DOM")

def sld_clean(val):
    if pd.isna(val) or str(val).strip().lower() in ("","nan","none",NOT_A_DNS): return "UNK_SLD"
    parts=str(val).strip().lower().lstrip(".").split(".")
    s=".".join(parts[-2:]) if len(parts)>=2 else parts[0]
    return s.upper() if s else "UNK_SLD"

def char_distribution_entropy(val):
    d=parse_dict_string(val)
    if not d: return 0.0
    total=sum(d.values())
    if total<=0: return 0.0
    return float(-sum((v/total)*math.log2(v/total) for v in d.values() if v>0))

def list_count(val): return len(parse_list_string(val))

def domain_ngram_tokens(val, top_k=5):
    items=parse_list_string(val)
    tokens=[str(t).strip().lower().replace(" ","_")
            for t in items[:top_k] if str(t).strip() and not str(t).strip().startswith(".")]
    return " ".join(tokens) if tokens else "UNK_NG"


# ===========================================================================
# Marques preprocessing
# ===========================================================================

def preprocess_marques(df: pd.DataFrame):
    """Full preprocessing for Marques et al.

    Deduplication note: the dataset contains many identical rows (it was built
    by replaying DNS queries from structured lists). We keep at most
    DEDUP_MAX_OCCURRENCES=5 copies of each identical row, consistent with the
    original paper's methodology and our submitted results (train size ~9,237).
    This is the correct and intended behaviour; the reviewer was informed of the
    resulting dataset size in the response letter.
    """
    df = df.copy()
    print("=== Preprocessing Marques ===")

    # 1. Detect and normalise label column (handles 'Class', ' Class', etc.)
    label_col = next(
        (c for c in df.columns if c.strip().lower() in ("class","label","target")),
        None
    )
    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")
    if label_col != TARGET_COL:
        df = df.rename(columns={label_col: TARGET_COL})
        print(f"  Renamed '{label_col}' → '{TARGET_COL}'")

    # Strip column name whitespace globally
    df.columns = [c.strip() for c in df.columns]

    # 2. Deduplication (max 5 identical rows — same as original notebook)
    before = len(df)
    df = df.groupby(list(df.columns), sort=False).head(config.DEDUP_MAX_OCCURRENCES)
    df = df.reset_index(drop=True)
    print(f"  [1/6] Dedup: {before-len(df):,} rows removed → {len(df):,} kept "
          f"(max {config.DEDUP_MAX_OCCURRENCES} identical rows per group)")

    # 3. Null strings → NaN
    df = fix_null_strings(df)

    # 4. Boolean strings → 0/1
    bool_cols = []
    for col in df.select_dtypes(include="object").columns:
        if col == TARGET_COL: continue
        uniq = set(df[col].dropna().astype(str).str.lower().unique())
        if uniq.issubset({"true","false","0","1","yes","no"}):
            bool_cols.append(col)
    if bool_cols:
        df = convert_bool_strings(df, bool_cols)

    # 5. Infinities
    df = fix_inf_values(df)

    # 6. Text columns
    str_cols  = df.select_dtypes(include="object").columns.tolist()
    text_cols = []
    for col in str_cols:
        if col in (TARGET_COL,): continue
        null_pct = df[col].isna().mean()
        n_unique = df[col].nunique()
        if null_pct < 0.50 and 2 <= n_unique <= 500:
            text_cols.append(col)
    if "DNSRecordType" in df.columns and "DNSRecordType" not in text_cols:
        text_cols.insert(0, "DNSRecordType")
    print(f"  [5/6] Text cols: {text_cols}")

    # 7. Impute
    df = impute_numeric(df)
    df = impute_categorical(df)

    # 8. Label normalise
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X_num, X_text, y, num_col_names = _split_features(df, text_cols, TARGET_COL)
    print(f"  X_num: {X_num.shape}  X_text: {len(X_text)}  y: {y.shape}")
    print(f"  Label dist: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X_num, X_text, y, num_col_names, text_cols


# ===========================================================================
# BCCC preprocessing
# ===========================================================================

def preprocess_bccc(df: pd.DataFrame):
    df = df.copy()
    print("=== Preprocessing BCCC ===")

    # 1. Filter non-DNS rows
    dns_marker_cols = [c for c in ["domain_name","top_level_domain","second_level_domain"]
                       if c in df.columns]
    if dns_marker_cols:
        before = len(df)
        mask = pd.Series(False, index=df.index)
        for col in dns_marker_cols:
            mask |= df[col].astype(str).str.strip().str.lower() == NOT_A_DNS
        df = df[~mask].reset_index(drop=True)
        print(f"  [1] Non-DNS rows removed: {before-len(df):,}")

    # 2-11. (same as before)
    df = fix_null_strings(df)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace({NOT_A_DNS: np.nan, "not a tcp connection": np.nan})

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
        df["bi_gram_tokens"] = df["bi_gram_domain_name"].apply(lambda v: domain_ngram_tokens(v,5))
        df = df.drop(columns=["bi_gram_domain_name"])
    for c in ["uni_gram_domain_name","tri_gram_domain_name"]:
        if c in df.columns: df = df.drop(columns=[c])
    if "character_distribution" in df.columns:
        df["char_dist_entropy"] = df["character_distribution"].apply(char_distribution_entropy)
        df["char_dist_size"]    = df["character_distribution"].apply(list_count)
        df = df.drop(columns=["character_distribution"])
    if "top_level_domain" in df.columns:
        df["tld_clean"] = df["top_level_domain"].apply(tld_clean)
        df = df.drop(columns=["top_level_domain"])
    if "second_level_domain" in df.columns:
        df["sld_clean"] = df["second_level_domain"].apply(sld_clean)
        df = df.drop(columns=["second_level_domain"])
    if "domain_name" in df.columns:
        df["domain_suffix"] = df["domain_name"].apply(domain_to_suffix)
        df = df.drop(columns=["domain_name"])

    KEEP_AS_TEXT = {"tld_clean","sld_clean","domain_suffix","query_rr_type_text",
                    "ans_rr_type_text","query_rr_class_text","ans_rr_class_text",
                    "bi_gram_tokens","attack_category"}
    for col in list(df.select_dtypes(include="object").columns):
        if col in KEEP_AS_TEXT or col=="label": continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() / max(df[col].notna().sum(),1) > 0.8:
            df[col] = coerced
        else:
            df = df.drop(columns=[col])

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c!="label"]
    variances = df[num_cols].replace([np.inf,-np.inf],np.nan).var()
    low_var   = variances[variances<1e-10].index.tolist()
    if low_var: df = df.drop(columns=low_var)

    df = fix_inf_values(df)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  Dedup: {before-len(df):,} rows removed")

    num_cols_log = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ("label",TARGET_COL)]
    for col in num_cols_log:
        s = df[col].dropna()
        if len(s)>0 and s.min()>=0 and s.skew()>3 and (s.max()-s.min())>100:
            df[col] = np.log1p(df[col].clip(lower=0))

    num_cols_corr = [c for c in df.select_dtypes(include=[np.number]).columns
                     if c not in ("label",TARGET_COL)]
    if len(num_cols_corr)>1:
        sample   = df[num_cols_corr].sample(n=min(50_000,len(df)),random_state=42)
        corr_mat = sample.corr().abs()
        upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape,dtype=bool),k=1))
        to_drop  = [c for c in upper.columns if any(upper[c]>0.95)]
        if to_drop: df = df.drop(columns=to_drop)

    df = impute_numeric(df)
    df = impute_categorical(df)
    num_final = df.select_dtypes(include=[np.number]).columns
    df[num_final] = df[num_final].fillna(0.0)

    # ── Extract category sidecar BEFORE label rename ──────────────────────
    # We keep it as a Series with the same index as df so it stays aligned.
    categories = None
    if "attack_category" in df.columns:
        categories = df["attack_category"].copy()
        df = df.drop(columns=["attack_category"])

    # Label
    df["label"] = pd.to_numeric(df["label"],errors="coerce").fillna(0).astype(int)
    df = df.rename(columns={"label": TARGET_COL})

    text_cols = [c for c in ["query_rr_type_text","ans_rr_type_text",
                              "query_rr_class_text","ans_rr_class_text",
                              "tld_clean","sld_clean","domain_suffix","bi_gram_tokens"]
                 if c in df.columns]

    X_num, X_text, y, num_col_names = _split_features(df, text_cols, TARGET_COL)
    print(f"  X_num: {X_num.shape}  X_text: {len(X_text)}  y: {y.shape}")
    return X_num, X_text, y, num_col_names, text_cols, categories


# ===========================================================================
# Feature split helpers
# ===========================================================================

def _build_text_sequence(df, text_cols):
    seq = None
    for col in text_cols:
        if col not in df.columns: continue
        part = df[col].fillna("UNK").astype(str).str.strip().str.upper()
        seq  = part if seq is None else seq + " " + part
    if seq is None:
        seq = pd.Series(["UNK"]*len(df), index=df.index)
    return seq

def _split_features(df, text_cols, target_col):
    y      = df[target_col].values.astype(int)
    X_text = _build_text_sequence(df, text_cols)
    drop   = [c for c in text_cols if c in df.columns] + [target_col]
    X_num  = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    return X_num.values.astype(np.float32), X_text, y, list(X_num.columns)


# ===========================================================================
# Train / Val / Test pipeline  — FIXED per-category alignment
# ===========================================================================

def full_pipeline(X_num: np.ndarray, X_text: pd.Series, y: np.ndarray,
                  dataset_name: str = "Dataset",
                  max_vocab: int = config.MAX_VOCAB_MARQUES,
                  max_seq: int   = config.MAX_SEQ_LEN,
                  categories     = None) -> dict:
    """Stratified 80/10/10 split + MinMaxScaler + Keras Tokenizer.

    Args:
        categories: optional Series or ndarray of category labels aligned
                    with (X_num, X_text, y). When provided, the test-set
                    slice is stored in data['y_te_categories'] so that
                    evaluate_per_category() works correctly.

    The categories array must have the SAME index / length as X_num / y.
    The split uses the same random_state=42 here and in evaluate, so the
    slicing is guaranteed to be aligned.
    """
    print(f"\n=== Pipeline {dataset_name} ===")

    n = len(y)
    all_idx = np.arange(n)

    # 1. Stratified 80 / 10 / 10
    idx_tr, idx_tmp = train_test_split(all_idx, test_size=(1-config.TRAIN_RATIO),
                                        random_state=42, stratify=y)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.5,
                                        random_state=42, stratify=y[idx_tmp])

    X_tr_n  = X_num[idx_tr];  X_val_n = X_num[idx_val]; X_te_n  = X_num[idx_te]
    X_tr_t  = X_text.iloc[idx_tr]; X_val_t = X_text.iloc[idx_val]; X_te_t = X_text.iloc[idx_te]
    y_tr    = y[idx_tr];  y_val = y[idx_val]; y_te = y[idx_te]

    print(f"  Split: {len(y_tr):,} train | {len(y_val):,} val | {len(y_te):,} test")

    # 2. MinMaxScaler — fit on TRAIN only
    for arr in (X_tr_n, X_val_n, X_te_n):
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    scaler  = MinMaxScaler()
    X_tr_n  = scaler.fit_transform(X_tr_n).astype(np.float32)
    X_val_n = scaler.transform(X_val_n).astype(np.float32)
    X_te_n  = scaler.transform(X_te_n).astype(np.float32)
    for arr in (X_tr_n, X_val_n, X_te_n):
        np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0, copy=False)

    # 3. Tokenizer — fit on TRAIN only
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_tr_t.tolist())
    vocab_size = min(len(tokenizer.word_index)+1, max_vocab)
    print(f"  Vocab: {len(tokenizer.word_index)} tokens → using {vocab_size}")

    def _tok(texts):
        seqs = tokenizer.texts_to_sequences(texts.tolist())
        return pad_sequences(seqs, maxlen=max_seq, padding="post",
                             truncating="post").astype(np.int32)

    # 4. Per-category slice — aligned via idx_te
    y_te_categories = None
    if categories is not None:
        if isinstance(categories, pd.Series):
            cats_arr = categories.values
        else:
            cats_arr = np.asarray(categories)
        if len(cats_arr) == n:
            y_te_categories = cats_arr[idx_te]
            print(f"  Categories: {len(np.unique(y_te_categories))} unique in test set")
        else:
            print(f"  [WARN] categories length {len(cats_arr)} != n {n}, skipping")

    return {
        "X_tr_num":  X_tr_n,  "X_val_num":  X_val_n,  "X_te_num":  X_te_n,
        "X_tr_seq":  _tok(X_tr_t), "X_val_seq": _tok(X_val_t), "X_te_seq": _tok(X_te_t),
        "y_tr": y_tr, "y_val": y_val, "y_te": y_te,
        "y_te_categories": y_te_categories,
        "scaler": scaler, "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "n_num_features": X_tr_n.shape[1],
        "max_seq": max_seq,
    }
