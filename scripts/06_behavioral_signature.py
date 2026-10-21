"""06 — Behavioural signature and emerging threats.

1. Profiles the fraud clusters found by DBSCAN and HDBSCAN. When both methods
   converge on the same group of transactions, that group is the fraud
   signature.
2. Looks at what One-Class SVM flagged beyond the confirmed frauds: possible
   new threats that have no labels yet.

Outputs:
  outputs/tables/06_fraud_clusters.csv
  outputs/tables/06_signature.csv
  outputs/tables/06_emerging_threats_by_currency.csv
"""

import os

import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

tx = pd.concat([pd.read_csv("data/transactions_clean.csv"),
                pd.read_csv("outputs/flags_unsupervised.csv")], axis=1)


def profile(group):
    return pd.Series({
        "transactions": len(group),
        "frauds": int(group["fraud"].sum()),
        "non_fraud": int((~group["fraud"]).sum()),
        "bin_freq": group["bin_freq"].median(),
        "ip_freq": group["ip_freq"].median(),
        "min_amount": group["amount"].min(),
        "max_amount": group["amount"].max(),
        "share_usd": group["is_usd"].mean(),
        "share_us": group["is_us"].mean(),
        "share_topup": group["is_topup"].mean(),
        "main_bin": group["bin_group"].mode()[0],
        "main_ip": group["ip_token"].mode()[0],
    })


# ---- 1. Fraud clusters ----------------------------------------------------

clusters = pd.concat([
    tx[tx["dbscan"]].groupby("dbscan_cluster").apply(profile, include_groups=False).assign(method="DBSCAN"),
    tx[tx["hdbscan"]].groupby("hdbscan_cluster").apply(profile, include_groups=False).assign(method="HDBSCAN"),
]).rename_axis("cluster").reset_index()
print("== Fraud clusters\n", clusters.to_string(index=False))
clusters.to_csv("outputs/tables/06_fraud_clusters.csv", index=False)

both = tx["dbscan"] & tx["hdbscan"]
print(f"\nDBSCAN and HDBSCAN agree on {both.sum()} transactions ({tx.loc[both, 'fraud'].sum()} frauds)")
print(f"HDBSCAN covers {tx.loc[tx['hdbscan'], 'fraud'].sum()} / {tx['fraud'].sum()} frauds")

# The signature: dominant card BIN group + IP of the cluster both methods share
signature = profile(tx[both])
print("\n== Signature\n", signature.to_string())
signature.to_frame("value").to_csv("outputs/tables/06_signature.csv")

# ---- 2. Emerging threats (One-Class SVM outliers that are not known frauds) --

emerging = tx[tx["ocsvm"] & ~tx["fraud"]]
print(f"\n== Emerging threats: {len(emerging)} transactions flagged by One-Class SVM with no fraud label")
print(f"First-time cards: {emerging['first_time_card'].sum()}")
print(f"IPs used by 3+ cards: {(emerging['cards_per_ip'] >= 3).sum()}")

print(f"Volume: EUR {emerging['amount_eur'].sum():,.0f} "
      f"(raw sum of mixed currencies: {emerging['amount'].sum():,.0f})")

by_currency = (
    emerging.groupby("currency")
    .agg(transactions=("amount", "size"), volume=("amount", "sum"),
         volume_eur=("amount_eur", "sum"), median_amount=("amount", "median"))
    .sort_values("volume_eur", ascending=False)
)
print(by_currency.head(10).round(2).to_string())
by_currency.to_csv("outputs/tables/06_emerging_threats_by_currency.csv")
