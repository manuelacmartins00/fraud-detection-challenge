"""04 — Supervised benchmark: Random Forest + SMOTE.

Upper bound for comparison with the unsupervised models. SMOTE runs inside
each cross-validation fold (imblearn Pipeline), so synthetic frauds never leak
into the fold being predicted. Every transaction is scored out-of-fold.

Outputs:
  outputs/flags_rf.csv
  outputs/tables/04_rf_feature_importance.csv
"""

import os

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEED = 42
os.makedirs("outputs/tables", exist_ok=True)

FEATURES = ["log_amount", "bin_freq", "ip_freq", "is_usd", "is_us", "is_topup", "approved"]

tx = pd.read_csv("data/transactions_clean.csv")
y = tx["fraud"].values
X = tx[FEATURES].astype(float)

pipe = Pipeline([
    ("smote", SMOTE(random_state=SEED)),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred = cross_val_predict(pipe, X, y, cv=cv).astype(bool)

pd.DataFrame({"rf_smote": pred}).to_csv("outputs/flags_rf.csv", index=False)
print(f"RF + SMOTE flagged {pred.sum()} transactions, {(pred & y).sum()} of {y.sum()} frauds")

# Feature importance from a model fitted on all data (interpretation only)
pipe.fit(X, y)
importance = (
    pd.Series(pipe.named_steps["rf"].feature_importances_, index=FEATURES, name="importance")
    .sort_values(ascending=False)
)
print("\nFeature importance:\n", importance.round(3).to_string())
importance.to_csv("outputs/tables/04_rf_feature_importance.csv")
