# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'
data = pd.read_excel(url)
data.columns = ['X1_Compactness', 'X2_SurfaceArea', 'X3_WallArea', 'X4_RoofArea', 
                'X5_Height', 'X6_Orientation', 'X7_GlazingArea', 'X8_GlazingDist', 
                'y1_HeatingLoad', 'y2_CoolingLoad']

# Features and Targets
X = data.iloc[:, :8].values
y = data.iloc[:, 8:].values

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data loading and preprocessing complete. Ready for model building.")
