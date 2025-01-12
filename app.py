from flask import Flask, render_template, request
import joblib
import tensorflow as tf
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load models
try:
    lgb_model = joblib.load("lightgbm_turnover_model.pkl")
except FileNotFoundError:
    print("LightGBM model file not found. Please ensure the model file is in the correct location.")
    lgb_model = None

try:
    lstm_model = tf.keras.models.load_model("lstm_turnover_model.h5", compile=False)
except FileNotFoundError:
    print("LSTM model file not found. Predictions using LSTM will not be available.")
    lstm_model = None

# List of regions from the training dataset
all_regions = ['Bayern', 'Niedersachsen', 'Nordrhein-westfalen', 'Baden-württemberg', 'Sachsen', 
               'Rheinland-pfalz', 'Schleswig-holstein', 'Hessen', 'Sachsen-anhalt', 'Saarland', 
               'Thüringen', 'Brandenburg', 'Mecklenburg-vorpommern', 'Berlin', 'Hamburg', 'Bremen']

# Preprocessing function for input features
def preprocess_input(store_area, avg_temp, precipitation, wind_speed, region, date):
    # Create a placeholder for one-hot encoding of regions
    region_encoded = [1 if reg == region else 0 for reg in all_regions]
    
    # Process the date (convert to numeric or extract features, e.g., day of the year)
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday  # Day of the year (1-365)
    
    # Combine all features into a single array
    return np.array([[store_area, avg_temp, precipitation, wind_speed, day_of_year] + region_encoded])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs
        store_area = float(request.form['store_area'])
        avg_temp = float(request.form['avg_temperature'])
        precipitation = float(request.form['precipitation_mm'])
        wind_speed = float(request.form['wind_speed_kmh'])
        region = request.form['region']
        date = request.form['date']
        model_choice = request.form['model_choice']

        # Preprocess inputs
        input_features = preprocess_input(store_area, avg_temp, precipitation, wind_speed, region, date)

        # Predict based on selected model
        if model_choice == 'LightGBM' and lgb_model is not None:
            prediction = lgb_model.predict(input_features, predict_disable_shape_check=True)[0]
        elif model_choice == 'LSTM' and lstm_model is not None:
            # Prepare sequence input for LSTM
            seq_length = 10
            lstm_input = np.expand_dims(np.tile(input_features, (seq_length, 1)), axis=0)
            prediction = lstm_model.predict(lstm_input)[0][0]
        else:
            prediction = "Invalid model choice or model file not found."

        # Pass the prediction to the results page
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
