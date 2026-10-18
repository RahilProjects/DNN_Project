# DNN_Project

A multi-target Deep Neural Network (DNN) to predict building heating and cooling loads for HVAC turbomachinery sizing, benchmarked against the UCI Energy Efficiency dataset.

## What is implemented

- Multi-output DNN regressor (`dnn_energy.py`) with:
  - ReLU hidden layers
  - Linear 2-target output (heating + cooling load)
  - Mini-batch gradient descent training
- Dataset loader for UCI Energy Efficiency tabular rows (8 input features + 2 targets)
- Holdout benchmarking pipeline with RMSE, MAE, and R² metrics per target
- CLI entrypoint for running benchmarks

## Dataset

Download the UCI Energy Efficiency data as a local text/CSV-like file with rows that contain:

`X1 X2 X3 X4 X5 X6 X7 X8 Y1 Y2`

Where:
- `Y1`: Heating Load
- `Y2`: Cooling Load

## Run benchmark

```bash
python dnn_energy.py --data /absolute/path/to/ENB2012_data.csv --epochs 300 --hidden 32,16
```

Example output:

```text
Benchmark metrics on holdout set:
Heating Load  -> RMSE: ...
Cooling Load  -> RMSE: ...
Average       -> RMSE: ...
```

## Run tests

```bash
python -m unittest -v test_dnn_energy.py
```
