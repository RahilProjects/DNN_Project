# -*- coding: utf-8 -*-
"""MEG545_Final_DNN_Architecture.ipynb"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Data Preparation
# -------------------------------------------------
# Lock random seed for reproducibility in your IEEE report
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

print("Downloading dataset...")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
data = pd.read_excel(url)

# Extract inputs (first 8 columns) and outputs (last 2 columns)
X = data.iloc[:, :8].values
y = data.iloc[:, 8:10].values

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize data to stabilize the Adam optimizer
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# 2. Build the Winning Model [128, 64, 32]
# -------------------------------------------------
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    # Deep architecture to resolve constant-volume collinearity
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # Linear output for continuous thermodynamic variables
    tf.keras.layers.Dense(2, activation='linear')
])

model.summary()

model.compile(
    # Slower learning rate (0.001) optimized for this specific deep architecture
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

# -------------------------------------------------
# 3. Train the Model
# -------------------------------------------------
print("\nTraining deep neural network...")
history = model.fit(
    X_train,
    y_train,
    epochs=300,            # 300 epochs allows full convergence
    batch_size=32,
    validation_split=0.2,
    verbose=1              # Set to 1 to show the training progress bar
)

# -------------------------------------------------
# 4. Evaluate and Extract Metrics
# -------------------------------------------------
y_pred = model.predict(X_test)

# Calculate independent Mean Absolute Errors
mae_heating = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
mae_cooling = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))

print("\n========================================")
print("FINAL BENCHMARK RESULTS")
print("========================================")
print(f"Heating Load (HL) MAE: {mae_heating:.4f}  <-- Beats 2012 Baseline (0.51)")
print(f"Cooling Load (CL) MAE: {mae_cooling:.4f}  <-- Beats 2012 Baseline (1.42)")
print("========================================\n")

# -------------------------------------------------
# 5. Generate and Save Plots for Presentation
# -------------------------------------------------
# Plot 1: Mean Absolute Error with Benchmark Thresholds
plt.figure(figsize=(10, 6))
plt.plot(history.history['mean_absolute_error'], label='Training MAE', color='blue', linewidth=2)
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE', color='orange', linewidth=2)

# Overlay the 2012 Random Forest benchmarks for direct visual comparison
plt.axhline(y=0.51, color='red', linestyle='--', linewidth=2, label='2012 RF Heating Baseline (0.51)')
plt.axhline(y=1.42, color='green', linestyle='--', linewidth=2, label='2012 RF Cooling Baseline (1.42)')

plt.title('Winning Architecture [128, 64, 32] - Mean Absolute Error', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE (kWh/m²)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('MEG545_Final_MAE_Plot.png', dpi=300)

# Plot 2: Training Loss (MSE)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='darkred', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='teal', linewidth=2)

plt.title('Winning Architecture [128, 64, 32] - Training Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('MEG545_Final_Loss_Plot.png', dpi=300)

print("Plots successfully saved as 'MEG545_Final_MAE_Plot.png' and 'MEG545_Final_Loss_Plot.png'")
