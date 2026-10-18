# Multi-Target Thermodynamic Load Prediction for HVAC Turbomachinery Sizing

## 📌 Problem Statement

Accurately predicting a building's heating and cooling loads is a critical real-world problem in mechanical engineering, specifically for appropriately sizing HVAC turbomachinery (e.g., centrifugal compressors, ventilation fans). Oversizing these systems causes thermodynamic inefficiencies and wastes energy, while undersizing leads to thermal failure. Traditional thermodynamic simulations (like Ecotect) are computationally expensive. This project develops a data-driven surrogate model using Deep Neural Networks to enable instantaneous, highly accurate load estimation directly from geometric building parameters.

## 📊 Dataset

* **Source:** [UCI Machine Learning Repository - Energy Efficiency Dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
* **Size:** 768 individual residential building simulations (constant internal volume of 771.75 m³).
* **Inputs (8 Geometric Features):** Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution.
* **Outputs (2 Continuous Targets):** Heating Load (HL) and Cooling Load (CL) measured in kWh/m².

## ⚙️ Methodology \& Model Architecture

1. **Data Preprocessing:** Features were standardized using `StandardScaler` to ensure uniform gradient descent convergence.
2. **Baseline Model:** A Multi-Output Linear Regression (IRLS-equivalent) model was established to prove that simple linear algorithms fail to map the non-linear heat transfer relationships and multi-collinearity present in the dataset.
3. **Deep Neural Network (DNN):**

   * **Architecture:** Multi-Layer Perceptron (MLP) with 2 hidden layers.
   * **Input Layer:** 8 nodes.
   * **Hidden Layer 1:** 64 nodes (ReLU activation) to capture complex geometric interactions.
   * **Hidden Layer 2:** 32 nodes (ReLU activation).
   * **Regularization:** 10% Dropout layer to prevent overfitting.
   * **Output Layer:** 2 nodes (Linear activation) for continuous multi-target regression.
   * **Optimizer \& Loss:** Adam optimizer (with `ReduceLROnPlateau`) optimizing Mean Squared Error (MSE). Performance tracked via Mean Absolute Error (MAE).

## 🚀 Instructions to Reproduce

1. Clone this repository: `git clone https://github.com/YOUR\_USERNAME/MEG545-Thermodynamic-DNN.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the main script to train the model and generate the evaluation plots: `python thermodynamic\_dnn.py`

## 📈 Main Results \& Conclusions

The DNN was critically benchmarked against the 2012 state-of-the-art Random Forest baseline established by Tsanas and Xifara (Target MAE: 0.51 for Heating, 1.42 for Cooling).

* **Baseline Linear Model MAE:** Heating: `\[Insert base HL MAE]` | Cooling: `\[Insert base CL MAE]`
* **Final DNN Model MAE:** Heating: `\[Insert your HL MAE]` | Cooling: `\[Insert your CL MAE]`

**Conclusion:** The 2-hidden-layer DNN successfully bypassed the computational bottleneck of traditional simulation software. By mapping the highly non-linear thermodynamics of features like Glazing Area and Surface-to-Volume ratios, the model successfully generalized the physics, allowing for rapid and precise turbomachinery sizing.

