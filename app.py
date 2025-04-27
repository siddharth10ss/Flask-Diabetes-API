from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

# Load the trained model and scaler
model = joblib.load("model/diabetes_rfr_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from form submission
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Scale the features using the saved scaler
        features_scaled = scaler.transform(features)

        # Predict using the loaded model
        prediction = model.predict(features_scaled)

        # Return the prediction result
        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
