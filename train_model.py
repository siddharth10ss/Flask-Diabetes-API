import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic data with more normal/healthy ranges"""
    np.random.seed(42)
    
    # Generate synthetic features with realistic distributions
    age = np.random.normal(45, 15, n_samples).clip(0, 100)  # Mean age 45, std 15
    sex = np.random.binomial(1, 0.5, n_samples)  # 50% male, 50% female
    bmi = np.random.normal(25, 5, n_samples).clip(10, 50)  # Mean BMI 25, std 5
    bp = np.random.normal(120, 20, n_samples).clip(50, 200)  # Mean BP 120, std 20
    
    # Generate S1-S6 with more normal distributions
    s1 = np.random.normal(150, 50, n_samples).clip(0, 300)
    s2 = np.random.normal(150, 50, n_samples).clip(0, 300)
    s3 = np.random.normal(150, 50, n_samples).clip(0, 300)
    s4 = np.random.normal(150, 50, n_samples).clip(0, 300)
    s5 = np.random.normal(150, 50, n_samples).clip(0, 300)
    s6 = np.random.normal(150, 50, n_samples).clip(0, 300)
    
    # Combine features
    X_synthetic = np.column_stack([age, sex, bmi, bp, s1, s2, s3, s4, s5, s6])
    
    # Generate target values with domain knowledge
    # Base progression on BMI, age, and blood pressure
    base_progression = (
        0.5 * (bmi - 25) +  # Further increased BMI contribution
        0.4 * (age - 45) +  # Further increased age contribution
        0.3 * (bp - 120) +  # Further increased BP contribution
        0.2 * (s1 - 150) +  # Increased S1 contribution
        0.2 * (s2 - 150) +  # Increased S2 contribution
        0.2 * (s3 - 150) +  # Increased S3 contribution
        0.1 * (s4 - 150) +  # Increased S4 contribution
        0.1 * (s5 - 150) +  # Increased S5 contribution
        0.1 * (s6 - 150)    # Increased S6 contribution
    )
    
    # Add some noise and ensure positive values
    y_synthetic = (base_progression + np.random.normal(0, 20, n_samples)).clip(0, 300)
    
    return X_synthetic, y_synthetic

# Load the original diabetes dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_original = diabetes.data
y_original = diabetes.target

# Generate synthetic data
X_synthetic, y_synthetic = generate_synthetic_data(n_samples=2000)

# Combine original and synthetic data
X_combined = np.vstack([X_original, X_synthetic])
y_combined = np.concatenate([y_original, y_synthetic])

# Create feature names
feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

# Create DataFrame for better visualization
df = pd.DataFrame(X_combined, columns=feature_names)
print("\nData shape:", X_combined.shape)
print("\nFeature statistics:")
print(df.describe())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining models...")

# Train Random Forest model with adjusted parameters
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,       # Further increased depth
    min_samples_split=8,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=42,
    oob_score=True,
    bootstrap=True
)
rf_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting model with adjusted parameters
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.15,  # Further increased learning rate
    max_depth=8,         # Further increased depth
    min_samples_split=8,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Evaluate both models
rf_pred = rf_model.predict(X_test_scaled)
gb_pred = gb_model.predict(X_test_scaled)

print("\nRandom Forest Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, rf_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, rf_pred):.2f}")
print(f"Out-of-bag Score: {rf_model.oob_score_:.2f}")

print("\nGradient Boosting Model Performance:")
print(f"Mean Squared Error: {mean_squared_error(y_test, gb_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, gb_pred):.2f}")

# Get feature importances for both models
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_model.feature_importances_
})
gb_importances = pd.DataFrame({
    'Feature': feature_names,
    'GB_Importance': gb_model.feature_importances_
})

importances = pd.merge(rf_importances, gb_importances, on='Feature')
importances['Average_Importance'] = (importances['RF_Importance'] + importances['GB_Importance']) / 2
importances = importances.sort_values('Average_Importance', ascending=False)

print("\nFeature Importances (Average of both models):")
print(importances)

# Calculate prediction ranges
rf_train_pred = rf_model.predict(X_train_scaled)
gb_train_pred = gb_model.predict(X_train_scaled)

print("\nPrediction Ranges:")
print("Random Forest:")
print(f"Min prediction: {min(rf_train_pred):.2f}")
print(f"Max prediction: {max(rf_train_pred):.2f}")
print(f"Mean prediction: {np.mean(rf_train_pred):.2f}")
print(f"Prediction std: {np.std(rf_train_pred):.2f}")

print("\nGradient Boosting:")
print(f"Min prediction: {min(gb_train_pred):.2f}")
print(f"Max prediction: {max(gb_train_pred):.2f}")
print(f"Mean prediction: {np.mean(gb_train_pred):.2f}")
print(f"Prediction std: {np.std(gb_train_pred):.2f}")

# Save the models and scaler
print("\nSaving models and scaler...")
os.makedirs('model', exist_ok=True)
joblib.dump(rf_model, 'model/diabetes_rf_model.pkl')
joblib.dump(gb_model, 'model/diabetes_gb_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Test the saved models
print("\nTesting saved models...")
loaded_rf = joblib.load('model/diabetes_rf_model.pkl')
loaded_gb = joblib.load('model/diabetes_gb_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')

# Test with high-risk input
test_input = np.array([[65, 1, 35, 180, 250, 250, 250, 250, 250, 250]])
test_input_scaled = loaded_scaler.transform(test_input)
rf_prediction = loaded_rf.predict(test_input_scaled)
gb_prediction = loaded_gb.predict(test_input_scaled)

print(f"\nTest predictions for high-risk input:")
print(f"Random Forest: {rf_prediction[0]:.2f}")
print(f"Gradient Boosting: {gb_prediction[0]:.2f}")
print(f"Ensemble Average: {(rf_prediction[0] + gb_prediction[0]) / 2:.2f}")

print("\nModels and scaler saved successfully!") 