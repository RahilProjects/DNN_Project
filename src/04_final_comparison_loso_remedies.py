"""Final comparison and leave-one-shape-out remedies on ENB2012.

0. Redraw the learning-curve and LOSO plots from the CSVs saved by 03_ablation_loso_learning_curve.py.
1. Final MLP ([256,128,64,32], lr 3e-3) vs random forest, XGBoost and linear regression on a new seed
   (10-fold CV x 3, identical folds for every model) with paired Wilcoxon tests.
2. Leave-one-shape-out CV with reduced geometry features (X1, X5, X6, X7, X8) and a seed ensemble.

Outputs go to ./results_v2. Set FAST = True for a quick smoke test. Approximate CPU runtime: 1 hour.
"""

# %% Imports & config
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

FAST = False
SEED = 2026                       # differs from the ablation seed (42), so the folds are new
N_SPLITS = 3 if FAST else 10
N_REPEATS = 1 if FAST else 3
ENS = 2 if FAST else 3            # seeds in the MLP ensemble
MAX_EPOCHS = 60 if FAST else 300
PATIENCE = 30
BATCH = 32
OLD = "results"                   # output folder of 03_ablation_loso_learning_curve.py
OUT = "results_v2"
os.makedirs(OUT, exist_ok=True)
TARGETS = ["HL", "CL"]
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

FINAL = dict(units=(256, 128, 64, 32), act="relu", lr=3e-3, l2=1e-4)
RED = [0, 4, 5, 6, 7]             # X1, X5, X6, X7, X8  (drop X2,X3,X4 = shape-derived geometry)

# %% Data
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
data = pd.read_excel(URL).dropna(how="all").iloc[:, :10]
X = data.iloc[:, :8].values.astype("float32")
y = data.iloc[:, 8:10].values.astype("float32")
groups = np.round(X[:, 0].astype("float64"), 2)
print("Data:", X.shape, y.shape, "| shapes:", len(np.unique(groups)))


# %% 0. Redraw learning-curve and LOSO plots from saved CSVs (skipped if the CSVs are missing)
def fix_old_plots():
    p = f"{OLD}/learning_curve.csv"
    if os.path.exists(p):
        lc = pd.read_csv(p)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
        for j, t in enumerate(TARGETS):
            for m in ["MLP", "RandomForest", "LinearReg"]:
                d = lc[(lc.target == t) & (lc.model == m)]
                g = d.groupby("frac").agg(n=("n_train", "mean"), mu=("MAE", "mean"),
                                          sd=("MAE", "std"), med=("MAE", "median"))
                ax[j].plot(g["n"], g["mu"], marker="o", label=f"{m} (mean)")
                ax[j].fill_between(g["n"], g["mu"] - g["sd"], g["mu"] + g["sd"], alpha=0.15)
            ax[j].set_xscale("log"); ax[j].set_yscale("log")
            ax[j].set_xlabel("# training samples"); ax[j].set_ylabel("MAE")
            ax[j].set_title(f"{t}: data efficiency"); ax[j].grid(alpha=0.4, ls=":"); ax[j].legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/C_learning_curve_fixed.png", dpi=300); plt.close()
    p = f"{OLD}/generalisation_results.csv"
    if os.path.exists(p):
        g = pd.read_csv(p)
        g = g[g.scheme == "leave-one-shape-out"].copy()
        g["held_out"] = g["held_out"].round(2)
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        for j, t in enumerate(TARGETS):
            s = g[g.target == t].groupby("model")["MAE"].agg(["mean", "median"])
            s.plot.bar(ax=ax[0, j], rot=0)
            ax[0, j].set_title(f"{t}: unseen-shape MAE (mean vs median)"); ax[0, j].grid(alpha=0.4, ls=":")
            pv = g[g.target == t].pivot_table(index="held_out", columns="model", values="MAE")
            pv.plot.bar(ax=ax[1, j], rot=45)
            ax[1, j].set_title(f"{t}: MAE per held-out shape"); ax[1, j].set_xlabel("held-out X1")
            ax[1, j].grid(alpha=0.4, ls=":")
        plt.tight_layout(); plt.savefig(f"{OUT}/B_generalisation_fixed.png", dpi=300); plt.close()
    print("[0] plots redrawn (if the CSVs were found)")


fix_old_plots()


# %% Models
def build_model(n_features, cfg):
    reg = tf.keras.regularizers.l2(cfg["l2"])
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for u in cfg["units"]:
        layers.append(tf.keras.layers.Dense(u, activation=cfg["act"], kernel_regularizer=reg))
    layers.append(tf.keras.layers.Dense(2, activation="linear"))
    m = tf.keras.models.Sequential(layers)
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["lr"]), loss="mse")
    return m


def fit_predict_mlp(X_tr, y_tr, X_te, cfg=FINAL, seed=SEED):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_s, X_te_s = sx.fit_transform(X_tr), sx.transform(X_te)
    y_tr_s = sy.fit_transform(y_tr)
    X_a, X_v, y_a, y_v = train_test_split(X_tr_s, y_tr_s, test_size=0.15, random_state=seed)
    model = build_model(X_tr.shape[1], cfg)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                patience=15, min_lr=1e-5)]
    model.fit(X_a, y_a, validation_data=(X_v, y_v), epochs=MAX_EPOCHS,
              batch_size=BATCH, callbacks=cbs, verbose=0)
    return sy.inverse_transform(model(X_te_s, training=False).numpy())


def fit_predict_lr(X_tr, y_tr, X_te):
    sx = StandardScaler()
    return LinearRegression().fit(sx.fit_transform(X_tr), y_tr).predict(sx.transform(X_te))


def fit_predict_rf(X_tr, y_tr, X_te):
    return RandomForestRegressor(n_estimators=300, random_state=SEED,
                                 n_jobs=-1).fit(X_tr, y_tr).predict(X_te)


try:
    from xgboost import XGBRegressor
    XGB_NAME = "XGBoost"

    def _xgb():
        return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                            subsample=0.8, colsample_bytree=0.9, random_state=SEED, n_jobs=-1)
except ImportError:                       # scikit-learn gradient boosting is used if xgboost is not installed
    from sklearn.ensemble import GradientBoostingRegressor
    XGB_NAME = "GradBoost(sklearn)"

    def _xgb():
        return GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, max_depth=4,
                                         subsample=0.8, random_state=SEED)
print("Boosting baseline:", XGB_NAME, "(fixed hyper-parameters, not tuned)")


def fit_predict_xgb(X_tr, y_tr, X_te):
    return MultiOutputRegressor(_xgb()).fit(X_tr, y_tr).predict(X_te)


def add_rows(rows, y_true, y_pred, **meta):
    for j, t in enumerate(TARGETS):
        e = y_true[:, j] - y_pred[:, j]
        rows.append(dict(target=t, **meta, MAE=np.mean(np.abs(e)), MSE=np.mean(e ** 2),
                         MRE=100 * np.mean(np.abs(e) / y_true[:, j]),
                         R2=r2_score(y_true[:, j], y_pred[:, j])))


# =====================================================================
# 1. FINAL MODEL on a new seed: 10-fold x 3 vs RF / XGBoost / LinearReg
# =====================================================================
splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X))
rows = []
t0 = time.time()
for i, (tr, te) in enumerate(splits):
    preds = {"MLP_final": fit_predict_mlp(X[tr], y[tr], X[te], FINAL, SEED + i),
             "RandomForest": fit_predict_rf(X[tr], y[tr], X[te]),
             XGB_NAME: fit_predict_xgb(X[tr], y[tr], X[te]),
             "LinearReg": fit_predict_lr(X[tr], y[tr], X[te])}
    for name, p in preds.items():
        add_rows(rows, y[te], p, model=name, fold=i)
    if (i + 1) % 5 == 0 or i == len(splits) - 1:
        pd.DataFrame(rows).to_csv(f"{OUT}/final_results.csv", index=False)
        print(f"[1] fold {i + 1}/{len(splits)}  ({time.time() - t0:.0f}s)")

fin = pd.DataFrame(rows)
fsum = fin.groupby(["target", "model"])[["MAE", "MSE", "MRE", "R2"]].agg(["mean", "std"]).round(3)
fsum.to_csv(f"{OUT}/final_summary.csv")
print("\n=== FINAL MODEL (new seed, 10-fold x 3) ===")
print(fsum)

print("\n=== Paired Wilcoxon on per-fold MAE (indicative: folds overlap) ===")
for t in TARGETS:
    a = fin[(fin.model == "MLP_final") & (fin.target == t)].sort_values("fold")["MAE"].values
    for other in ["RandomForest", XGB_NAME, "LinearReg"]:
        b = fin[(fin.model == other) & (fin.target == t)].sort_values("fold")["MAE"].values
        _, p = wilcoxon(a, b)
        print(f"{t}: MLP_final {a.mean():.3f} vs {other:18s} {b.mean():.3f} | p={p:.4f} "
              f"| MLP better in {(a < b).sum()}/{len(a)} folds")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
for j, t in enumerate(TARGETS):
    g = fin[fin.target == t].groupby("model")["MAE"].agg(["mean", "std"])
    ax[j].bar(g.index, g["mean"], yerr=g["std"], capsize=4)
    ax[j].axhline({"HL": 0.51, "CL": 1.42}[t], color="red", ls="--", label="Paper RF")
    ax[j].set_title(f"{t}: MAE, new-seed 10-fold x {N_REPEATS}")
    ax[j].tick_params(axis="x", rotation=20); ax[j].legend(); ax[j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/1_final_comparison.png", dpi=300); plt.close()

# =====================================================================
# 2. LEAVE-ONE-SHAPE-OUT remedies
# =====================================================================
logo = list(LeaveOneGroupOut().split(X, y, groups))
rows = []
t0 = time.time()
for i, (tr, te) in enumerate(logo):
    held = round(float(groups[te][0]), 2)
    Xtr_r, Xte_r = X[tr][:, RED], X[te][:, RED]
    red_preds = [fit_predict_mlp(Xtr_r, y[tr], Xte_r, FINAL, SEED + 100 * i + s) for s in range(ENS)]
    preds = {
        "MLP_all8": fit_predict_mlp(X[tr], y[tr], X[te], FINAL, SEED + i),
        "MLP_reduced": red_preds[0],
        f"MLP_reduced_ens{ENS}": np.mean(red_preds, axis=0),
        "RF_all8": fit_predict_rf(X[tr], y[tr], X[te]),
        "RF_reduced": fit_predict_rf(Xtr_r, y[tr], Xte_r),
        f"{XGB_NAME}_all8": fit_predict_xgb(X[tr], y[tr], X[te]),
        "LinearReg_all8": fit_predict_lr(X[tr], y[tr], X[te]),
        "LinearReg_reduced": fit_predict_lr(Xtr_r, y[tr], Xte_r),
    }
    for name, p in preds.items():
        add_rows(rows, y[te], p, model=name, held_out=held)
    pd.DataFrame(rows).to_csv(f"{OUT}/logo_results.csv", index=False)
    print(f"[2] shape {i + 1}/{len(logo)} (X1={held}) done  ({time.time() - t0:.0f}s)")

lg = pd.DataFrame(rows)
lsum = lg.groupby(["target", "model"])["MAE"].agg(["mean", "median", "std"]).round(3)
lsum.to_csv(f"{OUT}/logo_summary.csv")
print("\n=== LEAVE-ONE-SHAPE-OUT (MAE over 12 held-out shapes) ===")
print(lsum)

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for j, t in enumerate(TARGETS):
    s = lg[lg.target == t].groupby("model")["MAE"].agg(["mean", "median"])
    s.plot.bar(ax=ax[0, j], rot=30)
    ax[0, j].set_title(f"{t}: unseen-shape MAE (mean vs median)"); ax[0, j].grid(alpha=0.4, ls=":")
    keep = ["MLP_all8", f"MLP_reduced_ens{ENS}", "RF_all8", f"{XGB_NAME}_all8", "LinearReg_all8"]
    pv = lg[(lg.target == t) & (lg.model.isin(keep))].pivot_table(
        index="held_out", columns="model", values="MAE")
    pv.plot.bar(ax=ax[1, j], rot=45)
    ax[1, j].set_title(f"{t}: MAE per held-out shape"); ax[1, j].set_xlabel("held-out X1")
    ax[1, j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/2_logo_remedies.png", dpi=300); plt.close()

print("\nAll done. See ./results_v2")
