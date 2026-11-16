# Deep Surrogate Modelling of Residential Heating and Cooling Loads

ME G545 Deep Neural Networks, final project — Mohammad Rahil (2025H1410074P), BITS Pilani, Pilani Campus.

A multi-output MLP is studied as a fast surrogate for building-energy simulation on the ENB2012
dataset (768 simulated residential buildings, 8 inputs, 2 outputs: heating load HL and cooling load CL).
Beyond reporting accuracy, the project asks: **which design choices matter, how much data is needed,
and does the surrogate transfer to building shapes it has never seen?**

## Problem and dataset
- **Task:** regression, 8 geometric/glazing inputs -> HL and CL (kWh/m²).
- **Data:** UCI *Energy Efficiency* dataset (Tsanas & Xifara, 2012), 768 samples built from only
  **12 distinct building geometries**. Downloaded automatically by the scripts from
  <https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx>
  (not stored in this repo; CC BY 4.0).
- **Data issues handled:** X1/X2 perfectly rank-correlated, X4/X5 almost inversely related;
  inputs and targets are standardised using training folds only.

## Methodology
- **Model:** fully connected network, ReLU hidden layers, 2 linear outputs, MSE loss, Adam,
  batch 32, L2 = 1e-4, early stopping (15% inner validation split) + ReduceLROnPlateau.
  Final architecture `[256,128,64,32]`, learning rate `3e-3`.
- **Baselines:** linear regression, random forest, XGBoost (RF and XGBoost tuned by nested CV:
  randomised search inside each outer training fold, per target).
- **Experiments**
  1. One-factor-at-a-time study of 15 MLP configurations (architecture, activation, optimiser,
     learning rate, regularisation, feature encoding), 5-fold x 2 CV.
  2. Final comparison on a fresh seed, 10-fold x 3 CV, identical folds for all models, paired Wilcoxon tests.
  3. Generalisation: leave-one-shape-out (LOSO) CV, with a reduced-feature remedy and seed ensemble.
  4. Data-efficiency learning curve.

## Main results (10-fold x 3 CV, MAE in kWh/m²)
| Model | HL MAE | CL MAE |
|---|---|---|
| Linear regression | 2.092 | 2.267 |
| Random forest (tuned) | 0.314 | 0.991 |
| MLP (final) | 0.292 | 0.417 |
| XGBoost (tuned) | **0.192** | **0.361** |
| RF in original paper (Tsanas & Xifara) | 0.51 | 1.42 |

## Conclusions
- The tuned MLP is far better than linear regression and the originally published RF, but
  **tuned XGBoost is better on both targets** (paired test, 30 folds).
- Capacity and learning rate were the most influential MLP design choices.
- Deep networks need several hundred samples before they pay off (learning curve).
- On unseen building shapes (LOSO), all nonlinear models degrade by about an order of magnitude;
  a linear model with reduced features is best. Dropping shape-derived features (X2-X4) helps the MLP,
  but the evidence is only suggestive (paired p = 0.06-0.23 over 12 shapes).
- Random-CV accuracy therefore overstates how reliably a surrogate transfers to new geometries.

Limitations: simulated data, one climate, only 12 shapes; the MLP tuning was lighter than the
baselines' nested search; learning-curve and LOSO comparisons use untuned baselines.

## Repository structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_exploratory_data_analysis.py      # target histograms + Spearman correlation figure
│   ├── 02_baseline_cv_pipeline.py           # early repeated-CV pipeline: MLP vs linear regression vs RF
│   ├── 03_ablation_loso_learning_curve.py   # MLP ablation grid, LOSO generalisation, learning curve
│   ├── 04_final_comparison_loso_remedies.py # new-seed final comparison, LOSO remedies, redrawn plots
│   └── 05_nested_cv_baseline_tuning.py      # nested-CV tuning of RF and XGBoost
├── results/                                 # outputs of scripts 01-03 (CSV + PNG)
├── results_v2/                              # outputs of scripts 04-05 (CSV + PNG)
├── archive/                                 # first notebook (single 80/20 split), kept for history
└── report/                                  # IEEE report (.tex and .pdf)
```

## Reproducing the results
Python 3.10+ recommended. Run all commands from the repository root.
```bash
pip install -r requirements.txt
python src/01_exploratory_data_analysis.py     # a few seconds
python src/03_ablation_loso_learning_curve.py  # ~2 h on a CPU (set FAST = True at the top for a quick test)
python src/04_final_comparison_loso_remedies.py # ~1 h; reads results/ from the previous step to redraw plots
python src/05_nested_cv_baseline_tuning.py     # ~30 min; needs results_v2/final_results.csv from the previous step
```
`src/02_baseline_cv_pipeline.py` is an earlier, independent pipeline (MLP vs linear regression vs RF) and is not
needed for the main results.
Seeds: 42 (ablation, LOSO, learning curve) and 2026 (final comparison and tuned baselines).
Each script saves its tables and figures into `results/` or `results_v2/`.

## Acknowledgements and tools
Dataset: A. Tsanas and A. Xifara, UCI Machine Learning Repository (CC BY 4.0).
Libraries: TensorFlow/Keras, scikit-learn, XGBoost, pandas, NumPy, SciPy, Matplotlib.
No pretrained models were used.

## Reference
A. Tsanas and A. Xifara, "Accurate quantitative estimation of energy performance of residential
buildings using statistical machine learning tools," *Energy and Buildings*, vol. 49, pp. 560-567, 2012.
