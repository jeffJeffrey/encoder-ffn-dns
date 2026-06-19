# Encoder-FFN Model with Honeypot Deception for DNS Cache Poisoning Detection

**Reference paper:** Kengne Tchendji et al. (2026) — *Encoder-FFN Model with Honeypot Deception for DNS Cache Poisoning Detection and Mitigation in Software-Defined Network*

---

## Project structure

```
encoder_ffn_dns/
├── main_notebook.ipynb          # Orchestrator: runs the full pipeline end-to-end
├── requirements.txt
├── src/
│   ├── config.py                # Paths, hyper-parameters, seeds
│   ├── data_loader.py           # Automatic dataset download with cache
│   ├── preprocessing.py         # Cleaning, split, scaling, tokenisation
│   ├── eda.py                   # Exploratory data analysis plots
│   ├── train.py                 # Training loop + callbacks
│   ├── evaluate.py              # Metrics, plots, ablation, McNemar
│   ├── utils.py                 # save_fig (PDF+PNG), save_json, latency
│   └── models/
│       ├── __init__.py
│       └── encoder_ffn.py       # Full model + FFN-Only + Encoder-Only variants
├── data/                        # Auto-populated by data_loader
├── figures/                     # All plots (PDF + PNG)
│   ├── eda/{marques,bccc}/
│   ├── training/{marques,bccc}/
│   ├── evaluation/{marques,bccc}/
│   └── ablation/{marques,bccc}/
├── models/                      # Saved Keras models + artefacts
└── results/                     # JSON metrics files
```

## Quick start (Google Colab)

1. Open `main_notebook.ipynb` in Colab.
2. The first cell clones this repository automatically.
3. Add your Kaggle credentials in the 🔑 Secrets panel:
   - `KAGGLE_USERNAME` + `KAGGLE_KEY` **or** `KAGGLE_API_TOKEN`
4. Run all cells (`Runtime → Run all`).

## Datasets

| Dataset | Source | Size |
|---------|--------|------|
| Marques et al. | [GitHub](https://github.com/vanilink/DNS-Poisoning-Detection) | ~90k rows, 34 features |
| BCCC-CIC-Bell-DNS-2024 | [Kaggle](https://www.kaggle.com/datasets/bcccdatasets/malicious-dns-and-attacks-bccc-cic-bell-dns-2024) | ~965k rows, 120 features |

## Model architecture

```
Numerical input (MinMax-scaled)
  → FFN(32, ReLU) × 2  ──────────────────────────┐
                                                   ├─ Concat[96]
Textual input                                      │
  → Embedding(64) → MHA(4 heads)                  │
  → LayerNorm → FFN_txt → LayerNorm               │
  → GlobalAvgPool1D  ────────────────────────────┘
  → Dropout(0.35) → Dense(64, ReLU) → Dropout(0.20)
  → Dense(32, ReLU) → Dense(1, sigmoid)
```

**Tokenisation:** Keras `Tokenizer` trained from scratch on DNS text fields
(domain names, record types). Vocabulary sizes: 800 (Marques), 2000 (BCCC).
Max sequence length: 16. Embedding dimension: 64 (random init, trained end-to-end).

## Ablation study

Three variants are trained per dataset:

| Variant | Branches | Rationale |
|---------|----------|-----------|
| FFN-Only | Numerical only | Prior-art baseline |
| Encoder-Only | Textual only | Text-only upper bound |
| **Full (paper)** | Both fused | **Proposed model** |

Statistical significance is assessed with McNemar's test (α = 0.05).
