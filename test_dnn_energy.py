import tempfile
import unittest
from pathlib import Path

from dnn_energy import MultiTargetDNN, Standardizer, load_energy_efficiency_dataset, regression_metrics


class TestEnergyDNN(unittest.TestCase):
    def test_load_energy_efficiency_dataset_parses_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "energy.csv"
            path.write_text(
                "0.9,500,300,120,3,2,0.1,4,30.5,33.2\n"
                "0.8,550,280,110,4,3,0.2,3,27.2,29.8\n",
                encoding="utf-8",
            )
            features, targets = load_energy_efficiency_dataset(path)

        self.assertEqual(len(features), 2)
        self.assertEqual(len(targets), 2)
        self.assertEqual(features[0][:2], [0.9, 500.0])
        self.assertEqual(targets[1], [27.2, 29.8])

    def test_multi_target_dnn_learns_synthetic_relationship(self) -> None:
        x = []
        y = []
        for i in range(1, 121):
            base = i / 120.0
            row = [base, base * 2, base * 3, base * 4, 1 - base, 1 + base, 2 - base, 2 + base]
            heat = 4.0 * row[0] + 2.0 * row[1] + 0.5 * row[4]
            cool = 1.5 * row[2] + 0.8 * row[3] + 0.3 * row[7]
            x.append(row)
            y.append([heat, cool])

        split = 100
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        x_scaler = Standardizer.fit(x_train)
        y_scaler = Standardizer.fit(y_train)
        x_train_s = x_scaler.transform(x_train)
        x_test_s = x_scaler.transform(x_test)
        y_train_s = y_scaler.transform(y_train)

        model = MultiTargetDNN(input_size=8, hidden_sizes=(16, 8), output_size=2, seed=7)
        model.fit(x_train_s, y_train_s, epochs=400, learning_rate=0.02, batch_size=16, seed=7)

        pred_s = model.predict(x_test_s)
        pred = y_scaler.inverse_transform(pred_s)
        metrics = regression_metrics(y_test, pred)

        self.assertLess(metrics["average_rmse"], 0.06)
        self.assertGreater(metrics["average_r2"], 0.98)


if __name__ == "__main__":
    unittest.main()
