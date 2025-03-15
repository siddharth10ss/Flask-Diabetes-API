from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load trained model and scaler
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
        # Get form data from request
        data = request.form['features']

        # Convert input to numpy array
        features = np.array([float(x) for x in data.split(',')]).reshape(1, -1)

        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)

        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)