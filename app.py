from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import os

# Initialize the Flask application
app = Flask(__name__)

# --- Load the saved model components ---
# These are loaded only once when the application starts
try:
    model = joblib.load('xgb_model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)['columns']
    print("Model and components loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run train_and_save_model.py first.")
    model = None

# --- Define the prediction endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500

    # Get the JSON data from the request
    json_data = request.get_json()
    if not json_data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        # Convert the JSON data into a pandas DataFrame
        # The column order is enforced by using model_columns
        input_df = pd.DataFrame(json_data, index=[0])
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Make a prediction
        prediction_numeric = model.predict(input_df)
        
        # Convert the numeric prediction back to the original string label
        prediction_label = label_encoder.inverse_transform(prediction_numeric)

        # Create the response
        response = {
            'prediction': prediction_label[0],
            'status': 'Normal' if 'Benign' in prediction_label[0] else 'Malicious'
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
