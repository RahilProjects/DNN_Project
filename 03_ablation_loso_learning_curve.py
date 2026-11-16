"""MLP experiments on ENB2012: design-choice ablation, generalisation to unseen shapes, data efficiency.

A. One-factor-at-a-time study of 15 MLP configurations (5-fold CV x 2), with linear-regression and
   random-forest baselines on the same folds and a paired Wilcoxon test (base MLP vs RF).
B. Random K-fold vs leave-one-shape-out (LOSO) CV, grouping samples by relative compactness X1.
C. Learning curve: error versus number of training samples.

Set FAST = True for a quick smoke test. Each section saves its CSV and PNG files into ./results, so a
partial run keeps what has finished. Approximate CPU runtime: 2 hours.
"""

# %% Imports & config
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import RepeatedKFold, KFold, LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FAST = False                       # True for a quick smoke test
SEED = 42
N_SPLITS = 3 if FAST else 5
N_REPEATS = 1 if FAST else 2
MAX_EPOCHS = 100 if FAST else 300
PATIENCE = 30
BATCH = 32
OUT = "results"
os.makedirs(OUT, exist_ok=True)

PAPER_RF = {"HL": {"MAE": 0.51}, "CL": {"MAE": 1.42}}
TARGETS = ["HL", "CL"]

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# %% Data
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
data = pd.read_excel(URL).dropna(how="all").iloc[:, :10]
X = data.iloc[:, :8].values.astype("float32")
y = data.iloc[:, 8:10].values.astype("float32")
groups = np.round(X[:, 0], 2)      # building shape id (relative compactness X1)
print("Data:", X.shape, y.shape, "| unique shapes:", len(np.unique(groups)))


# %% Feature encodings  (X6 = orientation idx 5, X8 = glazing distribution idx 7)
def make_features(X, mode):
    if mode == "raw":
        return X
    if mode == "drop_X2":          # X2 (surface area) is perfectly collinear with X1
        return np.delete(X, 1, axis=1)
    orient = pd.get_dummies(X[:, 5].astype(int)).values.astype("float32")
    if mode == "onehot_orient":
        return np.hstack([X[:, :5], orient, X[:, 6:8]])
    if mode == "onehot_orient_glaz":
        dist = pd.get_dummies(X[:, 7].astype(int)).values.astype("float32")
        return np.hstack([X[:, :5], orient, X[:, 6:7], dist])
    raise ValueError(mode)


# %% Models
BASE = dict(units=(128, 64, 32), act="relu", opt="adam", lr=1e-3,
            l2=1e-4, dropout=0.0, features="raw")


def make_optimizer(name, lr):
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(lr, weight_decay=1e-4)
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    raise ValueError(name)


def build_model(n_features, cfg):
    reg = tf.keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for u in cfg["units"]:
        layers.append(tf.keras.layers.Dense(u, activation=cfg["act"], kernel_regularizer=reg))
        if cfg["dropout"] > 0:
            layers.append(tf.keras.layers.Dropout(cfg["dropout"]))
    layers.append(tf.keras.layers.Dense(2, activation="linear"))
    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer=make_optimizer(cfg["opt"], cfg["lr"]), loss="mse")
    return model


def fit_predict_mlp(X_tr, y_tr, X_te, cfg=BASE, seed=SEED):
    """Scale X and y on TRAIN only; early stopping on an inner validation split."""
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
    pred = sy.inverse_transform(model(X_te_s, training=False).numpy())
    return pred, model.count_params()


def fit_predict_lr(X_tr, y_tr, X_te):
    sx = StandardScaler()
    return LinearRegression().fit(sx.fit_transform(X_tr), y_tr).predict(sx.transform(X_te))


def fit_predict_rf(X_tr, y_tr, X_te):
    return RandomForestRegressor(n_estimators=300, random_state=SEED,
                                 n_jobs=-1).fit(X_tr, y_tr).predict(X_te)


def metrics(y_true, y_pred):
    out = {}
    for j, t in enumerate(TARGETS):
        e = y_true[:, j] - y_pred[:, j]
        out[t] = dict(MAE=np.mean(np.abs(e)), MSE=np.mean(e ** 2),
                      MRE=100 * np.mean(np.abs(e) / y_true[:, j]),
                      R2=r2_score(y_true[:, j], y_pred[:, j]))
    return out


def add_rows(rows, y_true, y_pred, **meta):
    for t, m in metrics(y_true, y_pred).items():
        rows.append(dict(target=t, **meta, **m))


# =====================================================================
# A. EXPERIMENT GRID  (one factor changed at a time from BASE)
# =====================================================================
def cfg(**kw):
    c = dict(BASE); c.update(kw); return c


GRID = [
    ("base",        "base [128,64,32] relu adam",  cfg()),
    ("architecture", "arch [32]",                  cfg(units=(32,))),
    ("architecture", "arch [64,32]",               cfg(units=(64, 32))),
    ("architecture", "arch [256,128,64,32]",       cfg(units=(256, 128, 64, 32))),
    ("activation",  "act tanh",                    cfg(act="tanh")),
    ("activation",  "act silu",                    cfg(act="silu")),
    ("optimizer",   "opt adamw",                   cfg(opt="adamw", l2=0.0)),
    ("optimizer",   "opt sgd+mom (lr 1e-2)",       cfg(opt="sgd", lr=1e-2)),
    ("learning rate", "lr 3e-4",                   cfg(lr=3e-4)),
    ("learning rate", "lr 3e-3",                   cfg(lr=3e-3)),
    ("regularisation", "no L2",                    cfg(l2=0.0)),
    ("regularisation", "dropout 0.1",              cfg(dropout=0.1)),
    ("features",    "onehot orientation",          cfg(features="onehot_orient")),
    ("features",    "onehot orient+glaz dist",     cfg(features="onehot_orient_glaz")),
    ("features",    "drop X2 (collinear)",         cfg(features="drop_X2")),
]

splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                            random_state=SEED).split(X))
rows = []
for factor, label, c in GRID:
    t0 = time.time()
    Xf = make_features(X, c["features"])
    for i, (tr, te) in enumerate(splits):
        pred, npar = fit_predict_mlp(Xf[tr], y[tr], Xf[te], c, seed=SEED + i)
        add_rows(rows, y[te], pred, factor=factor, config=label, fold=i, n_params=npar)
    print(f"[grid] {label:30s} done in {time.time() - t0:5.0f}s")
    pd.DataFrame(rows).to_csv(f"{OUT}/grid_results.csv", index=False)

# baselines on the same folds
for i, (tr, te) in enumerate(splits):
    add_rows(rows, y[te], fit_predict_lr(X[tr], y[tr], X[te]),
             factor="baseline", config="LinearReg", fold=i, n_params=np.nan)
    add_rows(rows, y[te], fit_predict_rf(X[tr], y[tr], X[te]),
             factor="baseline", config="RandomForest", fold=i, n_params=np.nan)
grid = pd.DataFrame(rows)
grid.to_csv(f"{OUT}/grid_results.csv", index=False)

order = list(grid["config"].unique())
summ = grid.groupby(["target", "config"])[["MAE", "MSE", "MRE", "R2"]].agg(["mean", "std"]).round(3)
summ.to_csv(f"{OUT}/grid_summary.csv")
print("\n=== GRID SUMMARY ===")
print(summ)

# Wilcoxon: base MLP vs RF, paired by fold (folds are not fully independent -> indicative only)
base_lbl = GRID[0][1]
print("\n=== Paired Wilcoxon, base MLP vs RF (MAE per fold) ===")
for t in TARGETS:
    a = grid[(grid.config == base_lbl) & (grid.target == t)].sort_values("fold")["MAE"].values
    b = grid[(grid.config == "RandomForest") & (grid.target == t)].sort_values("fold")["MAE"].values
    stat, p = wilcoxon(a, b)
    print(f"{t}: MLP {a.mean():.3f} vs RF {b.mean():.3f} | p = {p:.4f} "
          f"| MLP better in {(a < b).sum()}/{len(a)} folds")

# plot grid
fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
for j, t in enumerate(TARGETS):
    g = grid[grid.target == t].groupby("config")["MAE"].agg(["mean", "std"]).loc[order]
    ax[j].barh(range(len(order)), g["mean"], xerr=g["std"], capsize=3)
    ax[j].axvline(PAPER_RF[t]["MAE"], color="red", ls="--", label="Paper RF")
    ax[j].set_title(f"{t} - MAE (mean ± std over folds)")
    ax[j].set_xlabel("MAE (kWh/m²)"); ax[j].grid(alpha=0.4, ls=":"); ax[j].legend()
ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order); ax[0].invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/A_grid_mae.png", dpi=300); plt.close()

# complexity plot (architecture family)
arch = grid[grid.factor.isin(["base", "architecture"])]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for j, t in enumerate(TARGETS):
    g = arch[arch.target == t].groupby("config").agg(p=("n_params", "mean"),
                                                     m=("MAE", "mean"), s=("MAE", "std")).sort_values("p")
    ax[j].errorbar(g["p"], g["m"], yerr=g["s"], marker="o", capsize=3)
    ax[j].set_xscale("log"); ax[j].set_xlabel("# parameters"); ax[j].set_ylabel("MAE")
    ax[j].set_title(f"{t}: model complexity vs error"); ax[j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/A_complexity.png", dpi=300); plt.close()

# =====================================================================
# B. GENERALISATION: random K-fold vs leave-one-shape-out (group by X1)
# =====================================================================
n_groups = len(np.unique(groups))
schemes = {
    "random": list(KFold(n_splits=n_groups, shuffle=True, random_state=SEED).split(X)),
    "leave-one-shape-out": list(LeaveOneGroupOut().split(X, y, groups)),
}
rows = []
for scheme, sp in schemes.items():
    for i, (tr, te) in enumerate(sp):
        held = float(groups[te][0]) if scheme != "random" else i
        pm, _ = fit_predict_mlp(X[tr], y[tr], X[te], BASE, seed=SEED + i)
        for name, pred in [("MLP", pm),
                           ("RandomForest", fit_predict_rf(X[tr], y[tr], X[te])),
                           ("LinearReg", fit_predict_lr(X[tr], y[tr], X[te]))]:
            add_rows(rows, y[te], pred, scheme=scheme, model=name, held_out=held)
    print(f"[generalisation] {scheme} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/generalisation_results.csv", index=False)

gen = pd.DataFrame(rows)
gsum = gen.groupby(["target", "scheme", "model"])[["MAE", "R2"]].agg(["mean", "std"]).round(3)
gsum.to_csv(f"{OUT}/generalisation_summary.csv")
print("\n=== GENERALISATION SUMMARY ===")
print(gsum)

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
for j, t in enumerate(TARGETS):
    s = gen[(gen.target == t)].groupby(["scheme", "model"])["MAE"].mean().unstack()
    s.plot.bar(ax=ax[0, j], rot=0); ax[0, j].set_title(f"{t}: random vs unseen-shape MAE")
    ax[0, j].set_ylabel("MAE"); ax[0, j].grid(alpha=0.4, ls=":")
    p = gen[(gen.target == t) & (gen.scheme == "leave-one-shape-out")]
    p = p.pivot_table(index="held_out", columns="model", values="MAE")
    p.plot.bar(ax=ax[1, j], rot=45); ax[1, j].set_title(f"{t}: MAE per held-out shape (X1)")
    ax[1, j].set_xlabel("held-out relative compactness X1"); ax[1, j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/B_generalisation.png", dpi=300); plt.close()

# =====================================================================
# C. DATA-EFFICIENCY LEARNING CURVE
# =====================================================================
FRACS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
rng = np.random.default_rng(SEED)
rows = []
for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(X)):
    for f in FRACS:
        sub = rng.choice(tr, size=max(20, int(f * len(tr))), replace=False)
        pm, _ = fit_predict_mlp(X[sub], y[sub], X[te], BASE, seed=SEED + i)
        for name, pred in [("MLP", pm),
                           ("RandomForest", fit_predict_rf(X[sub], y[sub], X[te])),
                           ("LinearReg", fit_predict_lr(X[sub], y[sub], X[te]))]:
            add_rows(rows, y[te], pred, frac=f, n_train=len(sub), model=name, fold=i)
    print(f"[learning curve] fold {i + 1}/5 done")
    pd.DataFrame(rows).to_csv(f"{OUT}/learning_curve.csv", index=False)

lc = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for j, t in enumerate(TARGETS):
    for name in ["MLP", "RandomForest", "LinearReg"]:
        g = lc[(lc.target == t) & (lc.model == name)].groupby("n_train")["MAE"].agg(["mean", "std"])
        ax[j].plot(g.index, g["mean"], marker="o", label=name)
        ax[j].fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
    ax[j].axhline(PAPER_RF[t]["MAE"], color="red", ls="--", label="Paper RF")
    ax[j].set_xscale("log"); ax[j].set_yscale("log")
    ax[j].set_xlabel("# training samples"); ax[j].set_ylabel("MAE")
    ax[j].set_title(f"{t}: data efficiency"); ax[j].grid(alpha=0.4, ls=":"); ax[j].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/C_learning_curve.png", dpi=300); plt.close()

print("\nAll done. See the ./results folder (CSV + PNG).")
