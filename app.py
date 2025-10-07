import os
import joblib
import pandas as pd
import json
import numpy as np  # Import numpy
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# --- Load a pre-trained model, label encoder, and model columns ---
try:
    print("Attempting to load model and dependencies...")
    model = joblib.load('xgb_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)['columns']
    print("Model and dependencies loaded successfully.")
except Exception as e:
    print(f"FATAL: Error loading model files: {e}")

# --- Add a root endpoint for health checks and browser visits ---
@app.route('/', methods=['GET'])
def home():
    """A simple endpoint to confirm the API is running."""
    return "<h1>Network Intrusion Detection API</h1><p>The API is live. Please POST to the /predict endpoint.</p>"

# --- Use a specific /predict endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a POST request with JSON data for a single network flow,
    and returns a prediction.
    """
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Ensure all columns are numeric
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
        input_df.fillna(0, inplace=True)

        # --- FINAL, MOST ROBUST FIX ---
        # Convert the DataFrame to a NumPy array of type float32 before prediction
        # This prevents any potential dtype mismatches that can cause errors.
        prediction_data = np.array(input_df, dtype=np.float32)
        # --- END OF FIX ---

        try:
            # Pass the NumPy array to the model
            prediction_index = model.predict(prediction_data)
            prediction_label = label_encoder.inverse_transform(prediction_index)[0]
            return jsonify({'prediction': prediction_label})
        except Exception as pred_e:
            print(f"PREDICTION FAILED: {pred_e}")
            return jsonify({'error': f'Prediction failed: {pred_e}'}), 500

    except Exception as e:
        print(f"DATA HANDLING FAILED: {e}")
        return jsonify({'error': f'Data handling failed: {e}'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

