"""Exploratory data analysis of the ENB2012 energy-efficiency dataset.

Prints summary statistics and saves results/EDA.png: histograms of heating and cooling load and a
Spearman correlation matrix of the inputs and outputs. Run from the repository root.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
os.makedirs("results", exist_ok=True)

d = pd.read_excel(URL).dropna(how="all").iloc[:, :10]
d.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "HL", "CL"]
print(d.describe().round(3))
print("missing values:", int(d.isna().sum().sum()))
print("distinct X1 values (building shapes):", d["X1"].round(2).nunique())

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
ax[0].hist(d.HL, bins=30); ax[0].set_title("Heating load (kWh/m²)")
ax[1].hist(d.CL, bins=30); ax[1].set_title("Cooling load (kWh/m²)")
c = d.corr(method="spearman")
im = ax[2].imshow(c, cmap="coolwarm", vmin=-1, vmax=1)
ax[2].set_xticks(range(10)); ax[2].set_xticklabels(c.columns, rotation=90)
ax[2].set_yticks(range(10)); ax[2].set_yticklabels(c.columns)
ax[2].set_title("Spearman correlation"); fig.colorbar(im, ax=ax[2], fraction=0.046)
plt.tight_layout(); plt.savefig("results/EDA.png", dpi=300)
print("saved results/EDA.png")
