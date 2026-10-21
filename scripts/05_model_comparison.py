"""05 — Model comparison (Table 1 of the report).

Anomalies detected, frauds caught, recall, precision and F1 for every model.

Outputs:
  outputs/tables/05_model_comparison.csv
  outputs/figures/05_model_comparison.png
"""

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

MODELS = {
    "dbscan": "DBSCAN",
    "hdbscan": "HDBSCAN",
    "ocsvm": "One-Class SVM",
    "isolation_forest": "Isolation Forest Advanced",
    "lof": "LOF",
    "rf_smote": "Random Forest + SMOTE",
}

y = pd.read_csv("data/transactions_clean.csv")["fraud"].values
flags = pd.concat([pd.read_csv("outputs/flags_unsupervised.csv"),
                   pd.read_csv("outputs/flags_rf.csv")], axis=1)

rows = []
for col, name in MODELS.items():
    flagged = flags[col].values.astype(bool)
    tp = int((flagged & y).sum())
    recall = tp / y.sum()
    precision = tp / flagged.sum() if flagged.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    rows.append({
        "model": name,
        "anomalies_detected": int(flagged.sum()),
        "frauds_detected": tp,
        "recall": round(100 * recall, 2),
        "precision": round(100 * precision, 2),
        "f1": round(100 * f1, 2),
    })

table = pd.DataFrame(rows)
print(table.to_string(index=False))
table.to_csv("outputs/tables/05_model_comparison.csv", index=False)

ax = table.set_index("model")[["recall", "precision", "f1"]].plot.barh(
    figsize=(8, 4.5), color=["#13386B", "#00C2C7", "#9AA5B1"])
ax.invert_yaxis()
ax.set_xlabel("%")
ax.set_ylabel("")
ax.set_title("Fraud detection by model")
plt.tight_layout()
plt.savefig("outputs/figures/05_model_comparison.png", dpi=150)
