import pandas as pd
import numpy as np
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime
import os

# Load the model from MLFlow
model_uri = "runs:/97f637063d7348ffa423b11fe168b3c2/Random Forest Model"
model = mlflow.sklearn.load_model(model_uri)

# Streamlit application
st.title('Ames Housing Price Prediction')

st.write("""
### Enter the property details to predict the house price
""")

# Define input fields
def user_input_features():
    data = {
        'OverallQual': st.slider('OverallQual', 1, 10, 5),
        'GrLivArea': st.number_input('GrLivArea', min_value=334, max_value=5642, value=1500),
        'GarageCars': st.slider('GarageCars', 0, 4, 1),
        'GarageArea': st.number_input('GarageArea', min_value=0, max_value=1418, value=500),
        'TotalBsmtSF': st.number_input('TotalBsmtSF', min_value=0, max_value=6110, value=1000),
        '1stFlrSF': st.number_input('1stFlrSF', min_value=334, max_value=4692, value=1200),
        'FullBath': st.slider('FullBath', 0, 3, 2),
        'TotRmsAbvGrd': st.slider('TotRmsAbvGrd', 2, 14, 6),
        'YearBuilt': st.slider('YearBuilt', 1872, 2010, 1970)
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user inputs
st.subheader('User Input Features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.subheader('Predicted Sale Price')
    st.write(prediction)

    # Log prediction and actual values
    with open('predictions_log.csv', 'a') as f:
        input_df['Prediction'] = prediction
        input_df['Timestamp'] = datetime.datetime.now()
        input_df.to_csv(f, header=False, index=False)
        
    # For demo purposes, we assume user knows the actual sale price.
    #actual_price = st.number_input('Actual Sale Price', min_value=0, value=int(prediction[0]))
actual_price = st.number_input('Actual Sale Price', min_value=0)
if st.button('Log Actual Price'):
    print("actual_price",actual_price)
    with open('actuals_log.csv', 'a') as f:
        actual_log = pd.DataFrame({
            'Timestamp': [datetime.datetime.now()],
            'Actual': [actual_price]
        })
        actual_log.to_csv(f, header=False, index=False)
        print("done")

# Function to detect data drift
def detect_data_drift(reference_data, new_data, threshold=0.05):
    drifts = {}
    if new_data.empty:
        return drifts
    for column in reference_data.columns:
        if reference_data[column].dtype in ['int64', 'float64']:
            ref_mean = reference_data[column].mean()
            new_mean = new_data[column].mean()
            drift = abs(ref_mean - new_mean) / ref_mean
            if drift > threshold:
                drifts[column] = drift
    return drifts

# Function to detect model drift
def detect_model_drift(predictions_log, actuals_log, threshold=0.05):
    if predictions_log.empty or actuals_log.empty:
        return 0, None
    
    # Join predictions and actuals on timestamp
    data = predictions_log.join(actuals_log.set_index('Timestamp'), on='Timestamp')
    data = data.dropna()  # Drop rows with missing values

    # Calculate mean squared error
    if data.empty:
        return 0, None
    rmse = np.sqrt(mean_squared_error(data['Actual'], data['Prediction']))
    
    # Load initial model performance
    initial_rmse = pd.read_csv('initial_model_performance.csv').iloc[0]['rmse']
    
    # Calculate drift
    drift = abs(initial_rmse - rmse) / initial_rmse
    return drift, rmse

# Function to retrain the model
def retrain_model():
    # Load training data
    training_data = pd.read_csv('reference_data.csv')
    X = training_data.drop('target', axis=1)
    y = training_data['target']
    
    # Retrain the model
    model = RandomForestRegressor()
    model.fit(X, y)
    
    # Log the new model in MLflow
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("mse", mean_squared_error(y, model.predict(X)))
        
    # Update the model URI
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "AmesHousingModel")

# Load historical data for drift detection
reference_data = pd.read_csv('reference_data.csv')  # Historical data
try:
    predictions_log = pd.read_csv('predictions_log.csv', names=reference_data.columns.tolist() + ['Prediction', 'Timestamp'])
    actuals_log = pd.read_csv('actuals_log.csv', names=['Timestamp', 'Actual'])
except FileNotFoundError:
    predictions_log = pd.DataFrame(columns=reference_data.columns.tolist() + ['Prediction', 'Timestamp'])
    actuals_log = pd.DataFrame(columns=['Timestamp', 'Actual'])

# Detect data drift
drifts = detect_data_drift(reference_data, predictions_log[reference_data.columns])

st.subheader('Data Drift Detection')
if drifts:
    st.write('Drift detected in the following features:')
    st.write(drifts)
else:
    st.write('No data drift detected.')

# Detect model drift
model_drift, current_mse = detect_model_drift(predictions_log, actuals_log)

st.subheader('Model Drift Detection')
if current_mse is not None:
    st.write(f'Current MSE: {current_mse}')
    if model_drift > 0.05:  # You can adjust the threshold
        st.write(f'Model drift detected: {model_drift}')
        retrain_model()
        st.write('Model retrained and updated.')
    else:
        st.write('No model drift detected.')
else:
    st.write('Insufficient data to detect model drift.')
