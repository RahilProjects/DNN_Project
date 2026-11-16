"""Nested-CV tuning of the XGBoost and random-forest baselines on ENB2012.

Outer loop: the same 10-fold x 3 splits (seed 2026) as 04_final_comparison_loso_remedies.py, so every fold
pairs with the MLP result in results_v2/final_results.csv. Inner loop: 3-fold randomised search on the
training part of each outer fold only, tuned separately for HL and CL, so test folds never influence tuning.

Outputs go to ./results_v2. Set FAST = True for a quick smoke test. Approximate CPU runtime: 30 minutes.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, randint, uniform, loguniform
from sklearn.model_selection import RepeatedKFold, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

FAST = False
SEED = 2026                                  # must equal the seed in script 04 so the folds match
N_SPLITS = 3 if FAST else 10
N_REPEATS = 1 if FAST else 3                 # with 1 repeat the first 10 folds still pair with the MLP results
N_ITER_XGB = 3 if FAST else 15
N_ITER_RF = 3 if FAST else 10
PREV = "results_v2/final_results.csv"
OUT = "results_v2"
os.makedirs(OUT, exist_ok=True)
TARGETS = ["HL", "CL"]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
data = pd.read_excel(URL).dropna(how="all").iloc[:, :10]
X = data.iloc[:, :8].values.astype("float32")
y = data.iloc[:, 8:10].values.astype("float32")

# ---------------------------------------------------------------- search spaces
XGB_SPACE = {
    "n_estimators": randint(200, 1000),
    "learning_rate": loguniform(0.02, 0.3),
    "max_depth": randint(3, 9),
    "min_child_weight": randint(1, 6),
    "subsample": uniform(0.6, 0.4),          # 0.6 - 1.0
    "colsample_bytree": uniform(0.6, 0.4),   # 0.6 - 1.0
    "reg_lambda": loguniform(0.1, 10),
}
RF_SPACE = {
    "n_estimators": randint(200, 600),
    "max_depth": [None, 8, 12, 16, 24],
    "min_samples_leaf": randint(1, 4),
    "max_features": uniform(0.5, 0.5),       # fraction 0.5 - 1.0
}


def make_xgb():
    return XGBRegressor(random_state=SEED, n_jobs=1)


def make_rf():
    return RandomForestRegressor(random_state=SEED, n_jobs=1)


def tune_predict(make_est, space, n_iter, X_tr, y_tr, X_te, seed):
    """Separate inner random search per target; returns predictions + chosen params."""
    preds, params = [], []
    for j in range(y_tr.shape[1]):
        s = RandomizedSearchCV(make_est(), space, n_iter=n_iter,
                               cv=KFold(3, shuffle=True, random_state=seed),
                               scoring="neg_mean_absolute_error",
                               random_state=seed, n_jobs=-1, refit=True)
        s.fit(X_tr, y_tr[:, j])
        preds.append(s.predict(X_te))
        params.append({k: (float(v) if isinstance(v, (np.floating, float)) else
                           (int(v) if isinstance(v, (np.integer, int)) else v))
                       for k, v in s.best_params_.items()})
    return np.column_stack(preds), params


def add_rows(rows, y_true, y_pred, **meta):
    for j, t in enumerate(TARGETS):
        e = y_true[:, j] - y_pred[:, j]
        rows.append(dict(target=t, **meta, MAE=np.mean(np.abs(e)), MSE=np.mean(e ** 2),
                         MRE=100 * np.mean(np.abs(e) / y_true[:, j]),
                         R2=r2_score(y_true[:, j], y_pred[:, j])))


# ---------------------------------------------------------------- nested CV
splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X))
rows, best_log = [], []
t0 = time.time()
for i, (tr, te) in enumerate(splits):
    for name, mk, sp, n_it in [("XGB_tuned", make_xgb, XGB_SPACE, N_ITER_XGB),
                               ("RF_tuned", make_rf, RF_SPACE, N_ITER_RF)]:
        pred, params = tune_predict(mk, sp, n_it, X[tr], y[tr], X[te], SEED + i)
        add_rows(rows, y[te], pred, model=name, fold=i)
        best_log.append(dict(model=name, fold=i, HL=json.dumps(params[0]), CL=json.dumps(params[1])))
    pd.DataFrame(rows).to_csv(f"{OUT}/tuned_baselines_results.csv", index=False)
    pd.DataFrame(best_log).to_csv(f"{OUT}/tuned_baselines_best_params.csv", index=False)
    print(f"fold {i + 1}/{len(splits)} done ({time.time() - t0:.0f}s)")

tuned = pd.DataFrame(rows)

# ---------------------------------------------------------------- compare with the MLP and untuned baselines
if os.path.exists(PREV):
    prev = pd.read_csv(PREV)
    prev = prev[prev.fold < len(splits)]
    prev = prev.replace({"XGBoost": "XGB_untuned", "RandomForest": "RF_untuned"})
    allr = pd.concat([prev, tuned], ignore_index=True)
else:
    print("WARNING: previous results not found - comparing tuned baselines only")
    allr = tuned

summ = allr.groupby(["target", "model"])[["MAE", "MSE", "MRE", "R2"]].agg(["mean", "std"]).round(3)
summ.to_csv(f"{OUT}/tuned_baselines_summary.csv")
print("\n=== TUNED vs UNTUNED baselines vs MLP (same folds) ===")
print(summ)

if "MLP_final" in set(allr.model):
    print("\n=== Paired Wilcoxon on per-fold MAE: MLP_final vs each model ===")
    for t in TARGETS:
        a = allr[(allr.model == "MLP_final") & (allr.target == t)].sort_values("fold")["MAE"].values
        for other in ["XGB_tuned", "RF_tuned", "XGB_untuned", "RF_untuned"]:
            b = allr[(allr.model == other) & (allr.target == t)].sort_values("fold")["MAE"].values
            if len(a) == len(b) and len(a) > 5:
                _, p = wilcoxon(a, b)
                print(f"{t}: MLP_final {a.mean():.3f} vs {other:12s} {b.mean():.3f} | p={p:.4f} "
                      f"| MLP better in {(a < b).sum()}/{len(a)} folds")

# ---------------------------------------------------------------- plot
order = [m for m in ["MLP_final", "XGB_untuned", "XGB_tuned", "RF_untuned", "RF_tuned", "LinearReg"]
         if m in set(allr.model)]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for j, t in enumerate(TARGETS):
    g = allr[allr.target == t].groupby("model")["MAE"].agg(["mean", "std"]).loc[order]
    ax[j].bar(g.index, g["mean"], yerr=g["std"], capsize=4)
    ax[j].axhline({"HL": 0.51, "CL": 1.42}[t], color="red", ls="--", label="Paper RF")
    ax[j].set_title(f"{t}: MAE, tuned vs untuned baselines")
    ax[j].tick_params(axis="x", rotation=25); ax[j].legend(); ax[j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/3_tuned_baselines.png", dpi=300); plt.close()
print("\nDone. See results_v2/ (tuned_baselines_*.csv, 3_tuned_baselines.png)")
