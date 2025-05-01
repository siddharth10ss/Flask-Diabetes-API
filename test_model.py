import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load("model/diabetes_rfr_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Print model information
print("Model type:", type(model))
print("Model parameters:", model.get_params())

# Define feature names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Create more diverse test data
test_data = [
    [50, 1, 25.0, 80, 100, 200, 50, 200, 150, 100],  # Normal case
    [30, 0, 22.0, 70, 90, 180, 45, 180, 130, 90],    # Low values
    [60, 1, 28.0, 90, 110, 220, 55, 220, 170, 110],  # High values
    [40, 1, 35.0, 100, 150, 250, 70, 250, 200, 150], # Very high values
    [25, 0, 18.0, 60, 80, 150, 40, 150, 100, 80]     # Very low values
]

# Convert to DataFrame with feature names
test_df = pd.DataFrame(test_data, columns=feature_names)

# Scale the test data
test_data_scaled = scaler.transform(test_df)

# Make predictions
predictions = model.predict(test_data_scaled)

# Print results
print("\nTest Results:")
for i, (data, pred) in enumerate(zip(test_data, predictions)):
    print(f"\nTest Case {i+1}:")
    print("Input:")
    for name, value in zip(feature_names, data):
        print(f"  {name}: {value}")
    print(f"Prediction: {pred:.2f}")

# Check feature importances
if hasattr(model, 'feature_importances_'):
    print("\nFeature Importances:")
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(importances)

# Check model's tree structure
if hasattr(model, 'estimators_'):
    print("\nNumber of trees in the forest:", len(model.estimators_))
    print("Average tree depth:", np.mean([tree.get_depth() for tree in model.estimators_])) 