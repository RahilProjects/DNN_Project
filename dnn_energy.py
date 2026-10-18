"""Multi-target DNN for UCI Energy Efficiency heating/cooling prediction."""

from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _relu(x: float) -> float:
    return x if x > 0.0 else 0.0


def _relu_grad(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float], mean: float) -> float:
    if not values:
        return 1.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    return std if std > 1e-12 else 1.0


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(col) for col in zip(*matrix)]


def load_energy_efficiency_dataset(path: str | Path) -> Tuple[List[List[float]], List[List[float]]]:
    """Load UCI Energy Efficiency data from a local CSV/TXT-like file.

    Expects at least 10 numeric values per row:
    - first 8 columns: building features
    - columns 9 and 10: heating and cooling loads
    """

    rows = Path(path).read_text(encoding="utf-8").splitlines()
    features: List[List[float]] = []
    targets: List[List[float]] = []

    for row in rows:
        stripped = row.strip()
        if not stripped:
            continue
        tokens = [t for t in re.split(r"[\s,;]+", stripped) if t]
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            continue
        if len(values) < 10:
            continue
        features.append(values[:8])
        targets.append(values[8:10])

    if not features:
        raise ValueError("No valid numeric rows found. Ensure file contains UCI Energy Efficiency data.")
    return features, targets


def train_test_split(
    x: Sequence[Sequence[float]],
    y: Sequence[Sequence[float]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    indices = list(range(len(x)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split_at = max(1, int(len(indices) * (1.0 - test_ratio)))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]
    if not test_idx:
        test_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    x_train = [list(x[i]) for i in train_idx]
    y_train = [list(y[i]) for i in train_idx]
    x_test = [list(x[i]) for i in test_idx]
    y_test = [list(y[i]) for i in test_idx]
    return x_train, x_test, y_train, y_test


@dataclass
class Standardizer:
    means: List[float]
    stds: List[float]

    @classmethod
    def fit(cls, data: Sequence[Sequence[float]]) -> "Standardizer":
        cols = _transpose(data)
        means = [_mean(col) for col in cols]
        stds = [_std(col, means[idx]) for idx, col in enumerate(cols)]
        return cls(means=means, stds=stds)

    def transform(self, data: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [(row[i] - self.means[i]) / self.stds[i] for i in range(len(self.means))]
            for row in data
        ]

    def inverse_transform(self, data: Sequence[Sequence[float]]) -> List[List[float]]:
        return [
            [row[i] * self.stds[i] + self.means[i] for i in range(len(self.means))]
            for row in data
        ]


class MultiTargetDNN:
    """Simple feed-forward DNN with ReLU hidden layers and linear 2-target output."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (32, 16),
        output_size: int = 2,
        seed: int = 42,
    ) -> None:
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden layer sizes must be positive")

        self.layer_sizes = [input_size, *hidden_sizes, output_size]
        rnd = random.Random(seed)
        self.weights: List[List[List[float]]] = []
        self.biases: List[List[float]] = []

        for in_size, out_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = math.sqrt(2.0 / max(1, in_size))
            w = [[rnd.uniform(-scale, scale) for _ in range(out_size)] for _ in range(in_size)]
            b = [0.0 for _ in range(out_size)]
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, sample: Sequence[float]) -> Tuple[List[List[float]], List[List[float]]]:
        activations: List[List[float]] = [list(sample)]
        pre_activations: List[List[float]] = []

        current = list(sample)
        for layer_idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z: List[float] = []
            for j in range(len(b)):
                value = b[j]
                for i in range(len(current)):
                    value += current[i] * w[i][j]
                z.append(value)
            pre_activations.append(z)

            is_last_layer = layer_idx == len(self.weights) - 1
            current = z if is_last_layer else [_relu(v) for v in z]
            activations.append(current)

        return activations, pre_activations

    def predict(self, x: Sequence[Sequence[float]]) -> List[List[float]]:
        return [self._forward(sample)[0][-1] for sample in x]

    def fit(
        self,
        x: Sequence[Sequence[float]],
        y: Sequence[Sequence[float]],
        epochs: int = 300,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        if len(x) != len(y):
            raise ValueError("x and y must have the same number of samples")
        if not x:
            raise ValueError("training data is empty")

        rnd = random.Random(seed)
        n_samples = len(x)
        layer_count = len(self.weights)

        for _ in range(epochs):
            order = list(range(n_samples))
            rnd.shuffle(order)

            for batch_start in range(0, n_samples, batch_size):
                batch_ids = order[batch_start : batch_start + batch_size]
                if not batch_ids:
                    continue

                grad_w = [
                    [[0.0 for _ in range(len(self.weights[l][0]))] for _ in range(len(self.weights[l]))]
                    for l in range(layer_count)
                ]
                grad_b = [[0.0 for _ in range(len(self.biases[l]))] for l in range(layer_count)]

                for idx in batch_ids:
                    activations, pre_activations = self._forward(x[idx])
                    target = y[idx]

                    deltas: List[List[float]] = [[] for _ in range(layer_count)]
                    output = activations[-1]
                    output_delta = [2.0 * (output[j] - target[j]) / len(output) for j in range(len(output))]
                    deltas[-1] = output_delta

                    for l in range(layer_count - 2, -1, -1):
                        next_delta = deltas[l + 1]
                        next_weights = self.weights[l + 1]
                        current_delta: List[float] = []
                        for i in range(len(self.biases[l])):
                            backprop_sum = 0.0
                            for j in range(len(next_delta)):
                                backprop_sum += next_delta[j] * next_weights[i][j]
                            backprop_sum *= _relu_grad(pre_activations[l][i])
                            current_delta.append(backprop_sum)
                        deltas[l] = current_delta

                    for l in range(layer_count):
                        prev_activation = activations[l]
                        for i in range(len(prev_activation)):
                            for j in range(len(deltas[l])):
                                grad_w[l][i][j] += prev_activation[i] * deltas[l][j]
                        for j in range(len(deltas[l])):
                            grad_b[l][j] += deltas[l][j]

                batch_len = len(batch_ids)
                for l in range(layer_count):
                    for i in range(len(self.weights[l])):
                        for j in range(len(self.weights[l][i])):
                            self.weights[l][i][j] -= learning_rate * grad_w[l][i][j] / batch_len
                    for j in range(len(self.biases[l])):
                        self.biases[l][j] -= learning_rate * grad_b[l][j] / batch_len


def regression_metrics(y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred lengths must match")
    if not y_true:
        raise ValueError("cannot evaluate empty inputs")

    target_count = len(y_true[0])
    per_target = []
    for t in range(target_count):
        truths = [row[t] for row in y_true]
        preds = [row[t] for row in y_pred]
        errors = [preds[i] - truths[i] for i in range(len(truths))]
        mse = _mean([e * e for e in errors])
        rmse = math.sqrt(mse)
        mae = _mean([abs(e) for e in errors])
        truth_mean = _mean(truths)
        ss_res = sum((truths[i] - preds[i]) ** 2 for i in range(len(truths)))
        ss_tot = sum((truths[i] - truth_mean) ** 2 for i in range(len(truths)))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        per_target.append({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    return {
        "heating_load": per_target[0] if target_count > 0 else {},
        "cooling_load": per_target[1] if target_count > 1 else {},
        "average_rmse": _mean([m["rmse"] for m in per_target]),
        "average_mae": _mean([m["mae"] for m in per_target]),
        "average_r2": _mean([m["r2"] for m in per_target]),
    }


def benchmark_energy_dnn(
    dataset_path: str | Path,
    hidden_sizes: Sequence[int] = (32, 16),
    epochs: int = 300,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    x_raw, y_raw = load_energy_efficiency_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, seed=seed)

    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    x_test_s = x_scaler.transform(x_test)
    y_train_s = y_scaler.transform(y_train)

    model = MultiTargetDNN(
        input_size=len(x_train_s[0]),
        hidden_sizes=hidden_sizes,
        output_size=len(y_train_s[0]),
        seed=seed,
    )
    model.fit(
        x_train_s,
        y_train_s,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
    )

    y_pred_s = model.predict(x_test_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    return regression_metrics(y_test, y_pred)


def _parse_hidden_sizes(raw: str) -> Tuple[int, ...]:
    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        raise ValueError("hidden sizes must not be empty")
    return tuple(int(item) for item in items)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a multi-target DNN for UCI Energy Efficiency data "
            "(heating and cooling load prediction)."
        )
    )
    parser.add_argument("--data", required=True, help="Path to ENB2012 CSV/TXT-formatted data file.")
    parser.add_argument("--hidden", default="32,16", help="Comma-separated hidden layer sizes.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    metrics = benchmark_energy_dnn(
        dataset_path=args.data,
        hidden_sizes=_parse_hidden_sizes(args.hidden),
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("Benchmark metrics on holdout set:")
    print(f"Heating Load  -> RMSE: {metrics['heating_load']['rmse']:.4f}, MAE: {metrics['heating_load']['mae']:.4f}, R2: {metrics['heating_load']['r2']:.4f}")
    print(f"Cooling Load  -> RMSE: {metrics['cooling_load']['rmse']:.4f}, MAE: {metrics['cooling_load']['mae']:.4f}, R2: {metrics['cooling_load']['r2']:.4f}")
    print(f"Average       -> RMSE: {metrics['average_rmse']:.4f}, MAE: {metrics['average_mae']:.4f}, R2: {metrics['average_r2']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
