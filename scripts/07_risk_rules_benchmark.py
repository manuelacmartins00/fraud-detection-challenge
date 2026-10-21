"""07 — Benchmark against the existing defences.

How many transactions with the fraud signature (06) were already blocked by:
- the issuer's fraud check (ISSUER_SUSPECTED_FRAUD), and
- the merchant's own risk rules (BLOCKED BY MERCHANT RISK RULES)?

Output: outputs/tables/07_risk_rules_benchmark.csv
"""

import os

import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

tx = pd.read_csv("data/transactions_clean.csv")
signature = pd.read_csv("outputs/tables/06_signature.csv", index_col=0)["value"]

tx["has_signature"] = (tx["bin_group"] == signature["main_bin"]) & (tx["ip_token"] == signature["main_ip"])
print(f"Signature: BIN {signature['main_bin']} + IP {signature['main_ip']}")
print(f"Transactions with the signature: {tx['has_signature'].sum()} "
      f"({tx.loc[tx['has_signature'], 'fraud'].sum()} confirmed frauds)\n")

defences = {
    "ISSUER_SUSPECTED_FRAUD": "Issuer (bank) fraud check",
    "BLOCKED BY MERCHANT RISK RULES": "Merchant risk rules",
}
benchmark = (
    tx[tx["acquirer_response"].isin(defences)]
    .groupby("acquirer_response")
    .agg(blocked=("has_signature", "size"), with_signature=("has_signature", "sum"))
    .rename(index=defences)
)
benchmark["share_with_signature"] = (benchmark["with_signature"] / benchmark["blocked"]).round(3)
print(benchmark.to_string())
benchmark.to_csv("outputs/tables/07_risk_rules_benchmark.csv")
