"""Baseline pipeline: MLP vs linear regression vs random forest on ENB2012.

All models are evaluated on identical repeated 10-fold CV splits (the protocol of Tsanas & Xifara, 2012)
and scored with MAE, MSE, MRE and R2 for heating load (HL) and cooling load (CL). Inputs and targets are
standardised using the training part of each fold only. Results go to ./results.
"""

# %% Imports & config
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)
N_SPLITS = 10        # same as Tsanas & Xifara (2012)
N_REPEATS = 3        # the original study used 100 repeats
UNITS = (128, 64, 32)
L2 = 1e-4

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

PAPER_RF = {"HL": {"MAE": 0.51, "MSE": 1.03, "MRE": 2.18},
            "CL": {"MAE": 1.42, "MSE": 6.59, "MRE": 4.62}}

# %% Data
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
data = pd.read_excel(URL).dropna(how="all").iloc[:, :10]
X = data.iloc[:, :8].values.astype("float32")
y = data.iloc[:, 8:10].values.astype("float32")
print("Data:", X.shape, y.shape)


# %% Model
def build_model(n_features=8, units=UNITS, l2=L2):
    reg = tf.keras.regularizers.l2(l2)
    layers = [tf.keras.layers.Input(shape=(n_features,))]
    for u in units:
        layers.append(tf.keras.layers.Dense(u, activation="relu", kernel_regularizer=reg))
    layers.append(tf.keras.layers.Dense(2, activation="linear"))  # HL, CL
    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def fit_predict_mlp(X_tr, y_tr, X_te):
    """Scale X and y using TRAIN data only, train with early stopping, return preds in original units."""
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_s, X_te_s = sx.fit_transform(X_tr), sx.transform(X_te)
    y_tr_s = sy.fit_transform(y_tr)

    # inner validation split (from training fold only) for early stopping
    X_a, X_v, y_a, y_v = train_test_split(X_tr_s, y_tr_s, test_size=0.15, random_state=SEED)

    model = build_model(X_tr.shape[1])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=40,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=15, min_lr=1e-5),
    ]
    hist = model.fit(X_a, y_a, validation_data=(X_v, y_v),
                     epochs=500, batch_size=32, callbacks=cbs, verbose=0)
    pred = sy.inverse_transform(model.predict(X_te_s, verbose=0))
    return pred, hist.history


def fit_predict_lr(X_tr, y_tr, X_te):
    sx = StandardScaler()
    m = LinearRegression().fit(sx.fit_transform(X_tr), y_tr)
    return m.predict(sx.transform(X_te)), None


def fit_predict_rf(X_tr, y_tr, X_te):
    m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    return m.predict(X_te), None


MODELS = {"MLP": fit_predict_mlp, "LinearReg": fit_predict_lr, "RandomForest": fit_predict_rf}


# %% Metrics
def metrics(y_true, y_pred):
    out = {}
    for j, name in enumerate(["HL", "CL"]):
        e = y_true[:, j] - y_pred[:, j]
        out[name] = {
            "MAE": np.mean(np.abs(e)),
            "MSE": np.mean(e ** 2),
            "MRE": 100 * np.mean(np.abs(e) / y_true[:, j]),
            "R2": r2_score(y_true[:, j], y_pred[:, j]),
        }
    return out


# %% Repeated K-fold CV (same protocol for every model)
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
rows = []
first_history = None
oof_pred = {m: np.zeros_like(y) for m in MODELS}   # out-of-fold preds, repeat 0 only

for i, (tr, te) in enumerate(rkf.split(X)):
    for name, fn in MODELS.items():
        pred, hist = fn(X[tr], y[tr], X[te])
        if name == "MLP" and first_history is None:
            first_history = hist
        if i < N_SPLITS:
            oof_pred[name][te] = pred
        for target, vals in metrics(y[te], pred).items():
            rows.append({"model": name, "target": target, "fold": i, **vals})
    print(f"fold {i + 1}/{N_SPLITS * N_REPEATS} done")

res = pd.DataFrame(rows)
summary = res.groupby(["target", "model"])[["MAE", "MSE", "MRE", "R2"]].agg(["mean", "std"]).round(3)
print("\n=========== Out-of-sample results (mean, std over folds) ===========")
print(summary)
summary.to_csv(f"{OUT}/baseline_cv_results.csv")

# %% Comparison with the random-forest results of the 2012 study
print("\n=========== MLP vs 2012 paper RF (MAE) ===========")
for t in ["HL", "CL"]:
    r = res[(res.model == "MLP") & (res.target == t)]["MAE"]
    mean, std = r.mean(), r.std()
    base = PAPER_RF[t]["MAE"]
    verdict = "BEATS" if mean < base else "does NOT beat"
    print(f"{t}: MLP MAE = {mean:.3f} ± {std:.3f}  | paper RF = {base}  -> {verdict} baseline")

# %% Plots
# 1) learning curve (first fold; scaled-target MSE)
plt.figure(figsize=(8, 5))
plt.plot(first_history["loss"], label="Train loss", lw=2)
plt.plot(first_history["val_loss"], label="Validation loss", lw=2)
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("MSE (standardized targets)")
plt.title("MLP learning curve (fold 1, early stopping)")
plt.legend(); plt.grid(alpha=0.4, ls=":"); plt.tight_layout()
plt.savefig(f"{OUT}/baseline_learning_curve.png", dpi=300)

# 2) predicted vs actual, out-of-fold, per target
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for j, t in enumerate(["HL", "CL"]):
    ax[j].scatter(y[:, j], oof_pred["MLP"][:, j], s=12, alpha=0.6)
    lo, hi = y[:, j].min(), y[:, j].max()
    ax[j].plot([lo, hi], [lo, hi], "r--")
    ax[j].set_xlabel(f"Actual {t} (kWh/m²)"); ax[j].set_ylabel(f"Predicted {t}")
    ax[j].set_title(f"{t}: out-of-fold predictions")
    ax[j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/baseline_pred_vs_actual.png", dpi=300)

# 3) MAE comparison per model with paper RF reference
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for j, t in enumerate(["HL", "CL"]):
    g = res[res.target == t].groupby("model")["MAE"]
    ax[j].bar(g.mean().index, g.mean().values, yerr=g.std().values, capsize=4)
    ax[j].axhline(PAPER_RF[t]["MAE"], color="red", ls="--", label="Paper RF")
    ax[j].set_title(f"{t} MAE"); ax[j].legend(); ax[j].grid(alpha=0.4, ls=":")
plt.tight_layout(); plt.savefig(f"{OUT}/baseline_mae_comparison.png", dpi=300)
print("Saved plots and baseline_cv_results.csv to", OUT)
