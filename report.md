# Fraud Detection Report

**Author:** Manuel Martins · **Data:** 22,239 simulated payment transactions, 19–24 Nov 2025

This is the extended version of the 2-page report ([PDF](fraud_detection_report.pdf)).
It adds the full numbers, the reasoning behind each step, and recommendations.
Every number below is produced by the scripts in [`scripts/`](scripts/).

---

## The challenge

> Review a dataset of simulated payment transactions and prepare a short written
> analysis (maximum 2 pages, 48 hours):
>
> - Assess **payment performance**: how well payments are processed overall and where
>   results could be improved.
> - Analyse **disputed transactions** (potential fraud): their impact, where they appear
>   more often, and what that implies.
> - Conclude with the **next steps or further checks** you would recommend if this were
>   real company data.
>
> Any tools are allowed. The focus is on reasoning and communication, not just numbers.

---

## 1. Summary

- **Payments:** 77.75% of transactions are approved, for about **€453K** of volume (€368K approved).
  Most declines come from the issuer side (failed 3D Secure, insufficient funds, generic
  declines), not from the merchant's own rules.
- **Fraud:** 122 confirmed frauds/disputes, which is 0.71% of approved transactions and
  **$5,153 (≈ €4,467)** in losses. All of them are in USD, all on the same merchant
  account, and 98% are wallet top-ups.
- **The security checks don't catch it.** 99% of the frauds passed 3D Secure and carried
  liability shift, and none failed the CVC check.
- **Behaviour does catch it.** Clustering on velocity features (how often a card BIN and an
  IP repeat) isolates two card-testing rings that together hold 116 of the 122 frauds.
- **The merchant's risk rules missed both rings.** They blocked 884 transactions, but only 1
  of them matches the main fraud signature. The issuer's fraud check blocked 53.

## 2. Data

| | |
|---|---|
| Transactions | 22,239 (6 days) |
| Columns | 19: merchant account, payment method, currency, amount, shopper/issuer country, acquirer response, 3DS and CVC results, liability shift, app (domain), type, dispute fields, day, hashed card BIN group and IP token |
| Fraud label | `Record Type` present (NotificationOfFraud, Chargeback, NotificationOfChargeback): 122 cases |
| Currencies | 43. `Amount` is in the transaction's own currency, so all monetary totals here are converted to EUR at the rates of 21 Nov 2025 ([`fx_rates_eur.csv`](fx_rates_eur.csv)) |

**Cleaning ([`02_data_cleaning.R`](scripts/02_data_cleaning.R)).** No rows are removed.
- 5,849 rows are identical to another row. They are kept, because without a transaction ID
  or timestamp a repeated purchase can't be told apart from a duplicate.
- 166 zero-amount rows are all card verifications. They are flagged.
- Missing countries, types and apps are set to `UNKNOWN`.
- 3DS fields stay missing, because missing there means "not applicable".

## 3. Payment performance

| Acquirer response | Transactions | Share of declines | Volume (EUR) |
|---|---:|---:|---:|
| NOT_3D_AUTHENTICATED | 1,301 | 26.3% | 32.4K |
| BLOCKED BY MERCHANT RISK RULES | 884 | 17.9% | 16.3K |
| DECLINED | 797 | 16.1% | 9.6K |
| NOT_ENOUGH_BALANCE | 792 | 16.0% | 10.0K |
| ISSUER_SUSPECTED_FRAUD | 307 | 6.2% | 8.4K |
| *Other (16 reasons)* | 868 | 17.5% | 7.9K |

**Where approval can improve:**
- **3D Secure is the biggest single cause of declines** (26%). Transactions where the
  3DS directory returns `U` (unavailable) are approved only 10% of the time.
- **Weak segments.** Some segments with ≥100 transactions approve far below the 77.75%
  average: Bolivia (17%), Myanmar (17%), Thailand (55%), Mexico (58%), Maestro cards (57%)
  and METHOD_VERIFICATION transactions (35%).
- **PayPal approves 91%,** so offering local or alternative methods in those markets is the
  most direct lever.
- **The merchant's own rules block 18% of declines.** Their median blocked amount is €8.66
  and 58% are USD. Since they barely overlap with the confirmed frauds (section 6), part of
  these are probably legitimate customers turned away.

The approval rate is stable across the six days (75.6%–79.4%).

## 4. Fraud patterns

| Dimension | Concentration among the 122 frauds |
|---|---|
| Currency | 100% USD |
| Shopper country | 97% United States (3% Indonesia) |
| Issuer country | 99% United States |
| Payment method | 86% Visa, 14% Mastercard |
| Type / app | 98% `TOPUP` on the merchant's wallet app |
| Timing | 87 of 122 on 23 Nov. None on 19, 20 or 24 Nov |

**Amounts.** The mean fraud is $42.24 (≈ €36.6), against €20.3 for legitimate transactions.
The median is €8.7 for fraud and €4.3 for legitimate (Wilcoxon p < 0.001).

The mean hides two different groups:
- **Ring A:** 105 frauds on one card BIN group, almost all exactly **$10** (23 and 21 Nov).
- **Ring B:** 16 frauds on another BIN group, all exactly **$250**, all on 22 Nov.
  These account for **$4,000, or 78% of the money lost.**

**Security checks** ([`01_eda.R`](scripts/01_eda.R)). Among the frauds:

| Check | Frauds that passed |
|---|---:|
| 3D Secure (directory Y or A) | 99.2% |
| CVC not failed | 100% |
| Liability shift | 99.2% |

| Feature vs fraud | Cramér's V |
|---|---:|
| All security fields combined | 0.149 |
| 3DS directory × CVC | 0.135 |
| 3DS directory | 0.125 |
| 3DS authentication | 0.066 |
| Liability shift | 0.056 |
| CVC2 response | 0.029 |

Every association is statistically significant but weak (V ≤ 0.15). The fraudsters
are using cards that authenticate correctly, so detection has to rely on behaviour,
not on authentication results.

## 5. Machine learning

**Features.** Each transaction is described by:
- log amount
- card BIN frequency and IP frequency (velocity)
- whether it is USD, whether it is from the US, whether it is a top-up
- whether it was approved

Isolation Forest additionally uses cards per IP, IPs per card, first-time card, currency
frequency, country mismatch, liability shift and 3DS result.

**Models.**
- **DBSCAN and HDBSCAN** cluster the transactions without labels. A cluster is then marked as
  fraud when at least half of its members are confirmed frauds (cluster-then-label).
- **One-Class SVM, Isolation Forest and LOF** flag outliers without labels.
- **Random Forest + SMOTE** is the supervised upper bound. SMOTE runs inside each fold, and
  every transaction is predicted out-of-fold (5-fold stratified cross-validation).

| Model | Flagged | Frauds caught | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| DBSCAN | 119 | 100 | 81.97% | 84.03% | 82.99% |
| HDBSCAN | 140 | 116 | 95.08% | 82.86% | 88.55% |
| One-Class SVM | 379 | 99 | 81.15% | 26.12% | 39.52% |
| Isolation Forest (advanced) | 1,112 | 101 | 82.79% | 9.08% | 16.37% |
| LOF | 220 | 1 | 0.82% | 0.45% | 0.58% |
| Random Forest + SMOTE | 139 | 120 | 98.36% | 86.33% | 91.95% |

*The PDF shows the table from the original submission. DBSCAN and Random Forest + SMOTE
match it exactly. HDBSCAN (135 flagged), One-Class SVM (366 / 83), Isolation Forest
(1,112 / 84) and LOF (223 / 17) differ slightly because of hyperparameters. The ranking
and the conclusions are the same.*

**Reading the results:**
- **Density-based clustering works best without labels.** Fraud here is a dense group of
  near-identical transactions, which is exactly what DBSCAN and HDBSCAN find.
- **Point-anomaly detectors do poorly.** Isolation Forest and LOF look for isolated
  outliers, but this fraud is a crowd, not an outlier.
- **Random Forest + SMOTE should not be trusted as a production estimate.** With 122 frauds
  from essentially two rings, it learns those two rings. Its 98% recall says little about
  the next attack.
- **Robustness check.** Rerunning with amounts converted to EUR leaves DBSCAN and
  RF + SMOTE unchanged. HDBSCAN then finds only Ring A (100 frauds), and Isolation Forest
  gets worse.

## 6. Main findings

**Behavioural signature** ([`06_behavioral_signature.py`](scripts/06_behavioral_signature.py)).
DBSCAN and HDBSCAN converge on the same 119 transactions: 100 frauds plus 19 approved
transactions that are not labelled yet.

They all share:
- one card BIN group, seen 215 times in six days
- one IP, seen 189 times
- USD top-ups of $10–$20 from the US

That is automated card testing from a fixed location. HDBSCAN also isolates **Ring B**
(BIN seen 20 times, IP seen 18 times, $250 top-ups), bringing coverage to 116/122.

**The 19 unlabelled transactions in Ring A are the first thing to review.** They have the
same card group, IP and amounts as the frauds, and they were approved.

**Comparison with existing defences** ([`07_risk_rules_benchmark.py`](scripts/07_risk_rules_benchmark.py)).

| Defence | Blocked | With the Ring A signature |
|---|---:|---:|
| Issuer fraud check (ISSUER_SUSPECTED_FRAUD) | 307 | 53 |
| Merchant risk rules | 884 | 1 |

The issuers stopped part of the attack. The merchant's rules, which block almost three
times as many transactions, stopped one.

**Possible new threats.** One-Class SVM flags 280 transactions that have no fraud label.
- 176 of them are USD.
- 34 come from first-time cards.
- 66 come from IPs used by 3 or more cards.

They are worth about €7.8K in total. They are candidates for review, not confirmed fraud.

## 7. Recommendations

1. **Add velocity rules now.** For example: more than N transactions per card BIN or per IP
   in 24 hours, or repeated identical amounts on wallet top-ups. Both rings would have
   triggered after a handful of attempts.
2. **Review Ring A's 19 approved, unlabelled transactions** and refund or block as needed.
3. **Audit the merchant's risk rules.** 884 blocks with almost no overlap with confirmed
   fraud suggest they mostly stop legitimate customers. Measure their precision on a
   labelled sample and retire the rules that don't work.
4. **Score in real time.** Turn the clustering signal into an online score: rolling BIN and
   IP counts, cards per IP, first-time card plus a high amount. Send high scores to manual
   review rather than auto-declining.
5. **Improve 3D Secure handling.** Retry or route through another provider when the 3DS
   directory is unavailable (`U`), and test frictionless flows in the low-approval markets.
6. **Track performance by currency in EUR.** Report volumes converted, never as raw sums.

## 8. Next steps if this were real data

- **More history.** Use months of data, so models can be validated on later weeks than the
  ones they were trained on.
- **Real-time features.** Add timestamps, device fingerprint and account age, and compute
  velocity only from past transactions.
- **Feedback loop.** Feed chargebacks back as labels, because they arrive weeks later.
- **Measure in money.** Evaluate on cost (fraud losses + chargeback fees vs. lost revenue
  from false declines), not only recall and precision.
- **Test changes.** A/B test rule changes against approval rate and fraud rate together.

## 9. Reproducibility

```bash
Rscript scripts/01_eda.R
Rscript scripts/02_data_cleaning.R
python  scripts/03_unsupervised_models.py
python  scripts/04_random_forest_smote.py
python  scripts/05_model_comparison.py
python  scripts/06_behavioral_signature.py
python  scripts/07_risk_rules_benchmark.py
```

R 4.x with `dplyr`, `ggplot2`, `scales`. Python packages are in
[`requirements.txt`](requirements.txt). Exchange rates are mid-market rates of 2025-11-21
from the open [currency-api](https://github.com/fawazahmed0/exchange-api) dataset.
