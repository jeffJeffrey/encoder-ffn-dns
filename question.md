R1-C9 — Détails de la tokenisation

"How were domain names tokenised? Which tokenizer? Pre-trained or from scratch?"

Réponse :

We use the Keras Tokenizer class trained from scratch on the DNS text fields of the training split only (no pre-trained embeddings). The tokenizer operates at the word/subword level on space-separated DNS tokens (domain suffixes, record types, TLD codes). Key parameters: vocabulary size 800 for Marques (covering the full DNS token space with minimal sparsity), 2000 for BCCC (richer attack categories); OOV token <OOV>; maximum sequence length 16 (covers >99% of DNS response text fields without padding overhead). The embedding layer (dimension 64) is initialised randomly and trained end-to-end with the rest of the model. Larger vocabularies (2000, 5000) and dimensions (128, 256) were evaluated on the validation set of Marques but did not improve AUC while increasing training time.

R1-C11 / R2-C2 — Tests statistiques

"No statistical significance testing is provided."

Réponse :

We apply McNemar's test (with continuity correction for small disagreement counts) to compare the full model against each single-branch ablation variant on the held-out test set. McNemar's test is the standard non-parametric test for comparing two classifiers on the same test set (Dietterich, 1998). The contingency table counts the cases where one model is correct and the other is wrong. Results are reported as χ² statistic, p-value, and significance at α = 0.05.

R3-C1 — Détails d'entraînement (epochs, courbes, arrêt)

"More details on training behaviour (convergence, number of epochs) are needed."

Réponse :

Full training details are now reported in results/training_summary_marques.json and results/training_summary_bccc.json, including: total epochs run, best epoch (by val_loss), val_loss and val_accuracy at the best epoch, dataset sizes, batch size, tokenizer parameters, and embedding configuration. Training curves (loss, accuracy, precision/recall vs epoch) are saved as PDF in figures/training/{marques,bccc}/. Early stopping with patience=15 and ReduceLROnPlateau (factor=0.5, patience=5) are used on both datasets.

R3-C5 — Temps d'exécution, RAM, GPU

"Computational requirements (training time, RAM, GPU) are not reported."

Réponse :

Cell 15 of the notebook (training_summary) logs these values automatically. Indicative figures from our experiments on an NVIDIA T4 GPU (Google Colab): Marques — training time ~8 min (best epoch reached at ~35 epochs), peak RAM ~2 GB; BCCC — training time ~45 min (best epoch ~40 epochs), peak RAM ~6 GB. Inference latency is logged by measure_latency() (100 single-sample runs): sub-millisecond on both CPU and GPU, confirming suitability for real-time SDN deployment. These values are saved to results/training_summary_{dataset}.json.