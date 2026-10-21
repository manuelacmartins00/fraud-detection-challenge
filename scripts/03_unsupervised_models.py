"""03 — Unsupervised fraud detection.

Fits DBSCAN, HDBSCAN, One-Class SVM, Isolation Forest and LOF on the cleaned
data without using the fraud labels.

Clustering models are turned into flags with cluster-then-label: a cluster is
flagged as fraud when at least half of its members are confirmed frauds.
For the point-anomaly models (OCSVM, IF, LOF), every outlier is a flag.

Output: outputs/flags_unsupervised.csv (one row per transaction)
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
os.makedirs("outputs", exist_ok=True)

BASE = ["log_amount", "bin_freq", "ip_freq", "is_usd", "is_us", "is_topup", "approved"]
ADVANCED = BASE + ["amount", "cards_per_ip", "ips_per_card", "first_time_card",
                   "currency_freq", "country_mismatch", "liability_shift", "threeds_y"]

tx = pd.read_csv("data/transactions_clean.csv")
y = tx["fraud"].values
X = StandardScaler().fit_transform(tx[BASE].astype(float))
X_adv = StandardScaler().fit_transform(tx[ADVANCED].astype(float))


def flag_fraud_clusters(labels, min_fraud_share=0.5):
    share = pd.Series(y).groupby(labels).mean().drop(-1, errors="ignore")
    fraud_clusters = share.index[share >= min_fraud_share]
    print(f"  fraud clusters: {dict(share[fraud_clusters].round(3))}")
    return np.isin(labels, fraud_clusters)


flags = pd.DataFrame(index=tx.index)

print("DBSCAN")
flags["dbscan_cluster"] = DBSCAN(eps=0.5, min_samples=10).fit_predict(X)
flags["dbscan"] = flag_fraud_clusters(flags["dbscan_cluster"].values)

print("HDBSCAN")
flags["hdbscan_cluster"] = HDBSCAN(min_cluster_size=15, min_samples=15, copy=True).fit_predict(X)
flags["hdbscan"] = flag_fraud_clusters(flags["hdbscan_cluster"].values)

print("One-Class SVM")
flags["ocsvm"] = OneClassSVM(nu=0.01, gamma="scale").fit_predict(X) == -1

print("Isolation Forest (advanced features)")
iforest = IsolationForest(n_estimators=200, contamination=0.05, random_state=SEED)
flags["isolation_forest"] = iforest.fit_predict(X_adv) == -1

print("LOF")
flags["lof"] = LocalOutlierFactor(n_neighbors=20, contamination=0.01).fit_predict(X) == -1

flags.to_csv("outputs/flags_unsupervised.csv", index=False)
print("\nFlagged per model:")
print(flags[["dbscan", "hdbscan", "ocsvm", "isolation_forest", "lof"]].sum().to_string())
