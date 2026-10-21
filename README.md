# Payment Fraud Detection — Technical Challenge

Analysis of 22,239 simulated payment transactions (19–24 Nov 2025) from a fraud analyst
technical challenge: how well payments are processed, where disputed transactions
concentrate, and how to catch them.

- **[fraud_detection_report.pdf](fraud_detection_report.pdf)**: the 2-page report.
- **[report.md](report.md)**: the extended version, with the challenge brief, full numbers,
  reasoning and recommendations.
- **[scripts/](scripts/)**: the full pipeline, with R for EDA and cleaning and Python for the models.

## Key findings

- **Approval rate 77.75%.** The main decline reasons are failed 3D Secure (1,301),
  the merchant's own risk rules (884), generic declines (797) and insufficient funds (792).
- **122 confirmed frauds/disputes, worth $5,153.** All are in USD, 97% from the US and
  98% are wallet top-ups. They come from two rings: 105 repeated $10 top-ups, and
  16 top-ups of $250 that account for 78% of the money lost.
- **The security checks don't catch them.** 99% of the frauds passed 3D Secure and
  carry liability shift, and their CVC never failed. Every security feature has a weak
  association with fraud (Cramér's V ≤ 0.15).
- **Behaviour does catch them.** DBSCAN isolates one cluster that holds 100 of the 122
  frauds: one card BIN group seen 215 times and one IP seen 189 times, all making
  USD top-ups. That is automated card testing from a fixed location. HDBSCAN also finds
  the second ring and covers 116/122 (95%).
- **The existing defences don't overlap with the signature.** The issuer's fraud check
  had already blocked 53 transactions with this signature, while the merchant's risk
  rules blocked only 1 of their 884.

## Model comparison

| Model | Flagged | Frauds caught | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| DBSCAN | 119 | 100 | 81.97% | 84.03% | 82.99% |
| HDBSCAN | 140 | 116 | 95.08% | 82.86% | 88.55% |
| One-Class SVM | 379 | 99 | 81.15% | 26.12% | 39.52% |
| Isolation Forest (advanced features) | 1,112 | 101 | 82.79% | 9.08% | 16.37% |
| LOF | 220 | 1 | 0.82% | 0.45% | 0.58% |
| Random Forest + SMOTE (supervised, 5-fold out-of-fold) | 139 | 120 | 98.36% | 86.33% | 91.95% |

All the unsupervised models are fitted without the labels. For DBSCAN and HDBSCAN,
the labels are used only afterwards, to mark a cluster as fraud when at least half
of its members are confirmed frauds. Random Forest + SMOTE is the supervised upper
bound, and SMOTE is applied inside each fold only.

## Pipeline

| Step | Script | Language | What it does |
|---|---|---|---|
| 1. EDA & statistical testing | `scripts/01_eda.R` | R | Approval rate, decline reasons, daily trend, segments, fraud concentration, amounts (Wilcoxon), χ² / Cramér's V on the security checks |
| 2. Data cleaning | `scripts/02_data_cleaning.R` | R | Names, types, missing values, quality checks, currency conversion, labels and behavioural features, written to `data/transactions_clean.csv` |
| 3. Unsupervised models | `scripts/03_unsupervised_models.py` | Python | DBSCAN, HDBSCAN, One-Class SVM, Isolation Forest, LOF |
| 4. Supervised benchmark | `scripts/04_random_forest_smote.py` | Python | Random Forest + SMOTE with out-of-fold predictions, feature importance |
| 5. Model comparison | `scripts/05_model_comparison.py` | Python | Metrics table and chart |
| 6. Behavioural signature | `scripts/06_behavioral_signature.py` | Python | Fraud-cluster profiles, the signature, emerging threats from One-Class SVM |
| 7. Risk rules benchmark | `scripts/07_risk_rules_benchmark.py` | Python | How many signature transactions the issuer and the merchant's rules already blocked |

Tables are written to `outputs/tables/` and charts to `outputs/figures/`.

## Usage

Run everything from the project root:

```bash
# R packages
Rscript -e 'install.packages(c("dplyr", "ggplot2", "scales"))'
# Python packages
pip install -r requirements.txt

Rscript scripts/01_eda.R
Rscript scripts/02_data_cleaning.R
python  scripts/03_unsupervised_models.py
python  scripts/04_random_forest_smote.py
python  scripts/05_model_comparison.py
python  scripts/06_behavioral_signature.py
python  scripts/07_risk_rules_benchmark.py
```

## Data

[`fraud_challenge_dataset.csv`](fraud_challenge_dataset.csv) contains simulated transactions.
Merchant accounts, merchant-specific rule names and app identifiers are anonymised.
Card BINs and IPs were already provided as hashed group tokens.

## Notes and limitations

- **Small sample.** Six days of data and 122 confirmed frauds are not enough to train
  a model that would hold up in production.
- **Currencies.** `Amount` is in each transaction's own currency (43 currencies).
  Monetary totals are converted to EUR with [`fx_rates_eur.csv`](fx_rates_eur.csv)
  (mid-market rates of 2025-11-21). The models use the original amounts.
- **Velocity features are computed over the whole window.** BIN and IP frequencies use
  all six days. In production they would be rolling counts that only look at the past.
- **Two tables, slightly different numbers.** The table in the PDF comes from the original
  submission. DBSCAN and Random Forest + SMOTE match it exactly. HDBSCAN, One-Class SVM,
  Isolation Forest and LOF differ slightly in this pipeline because of hyperparameters.
  The ranking and the conclusions are the same.

## Tech stack

- **R:** dplyr, ggplot2, scales (EDA, statistical testing, cleaning)
- **Python:** pandas, scikit-learn, imbalanced-learn, matplotlib (anomaly detection, clustering, RF + SMOTE)

## Author

Manuel Martins
