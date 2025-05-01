import joblib
import numpy as np
import pandas as pd

# Load the scaler
scaler = joblib.load("model/scaler.pkl")

# Print scaler information
print("Scaler type:", type(scaler))
print("\nScaler mean values:")
for i, mean in enumerate(scaler.mean_):
    print(f"Feature {i}: {mean:.2f}")

print("\nScaler scale values (std dev):")
for i, scale in enumerate(scaler.scale_):
    print(f"Feature {i}: {scale:.2f}")

# Create example data points at the extremes
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
extreme_low = np.array([[20, 0, 18, 60, 80, 150, 40, 150, 100, 80]])
extreme_high = np.array([[70, 1, 40, 120, 200, 300, 80, 300, 250, 200]])

# Scale the extreme values
scaled_low = scaler.transform(extreme_low)
scaled_high = scaler.transform(extreme_high)

print("\nScaled values for extreme inputs:")
print("\nExtreme Low Input:")
for name, orig, scaled in zip(feature_names, extreme_low[0], scaled_low[0]):
    print(f"{name}: Original={orig:.1f}, Scaled={scaled:.2f}")

print("\nExtreme High Input:")
for name, orig, scaled in zip(feature_names, extreme_high[0], scaled_high[0]):
    print(f"{name}: Original={orig:.1f}, Scaled={scaled:.2f}") 