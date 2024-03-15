# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the pre-trained model
model_path = 'linear_regression_model.pkl'
model = joblib.load(model_path)

# Function to preprocess input data
def preprocess_input(year, present_price, selling_type, km_driven, fuel_type, transmission, owner, label_encoder, scaler):
    # Calculate car_age from the current year
    current_year = 2024
    car_age = current_year - year

    # Create a DataFrame with input data
    input_data = pd.DataFrame({
        'Present_Price': [present_price],
        'Selling_type': [selling_type],
        'Driven_kms': [km_driven],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Owner': [owner],
        'car_age': [car_age]
    })

    # Perform label encoding using the provided encoder
    input_data["Fuel_Type"] = label_encoder.transform(input_data["Fuel_Type"])
    input_data["Transmission"] = label_encoder.transform(input_data["Transmission"])
    input_data["Selling_type"] = label_encoder.transform(input_data["Selling_type"])

    # Perform feature scaling using the provided scaler
    scaled_data = scaler.transform(input_data)

    return scaled_data

# Streamlit app
def main():
    st.title('Car Price Prediction App')

    # Input form
    st.sidebar.header('User Input Features')

    # Get user input
    year = st.sidebar.slider('Year of Manufacture', 1990, 2023, 2010)
    present_price = st.sidebar.number_input('Present Price', min_value=1.0)
    selling_type = st.sidebar.selectbox('Selling Type', ('Individual', 'Dealer'))
    km_driven = st.sidebar.number_input('Kilometers Driven', min_value=1)
    fuel_type = st.sidebar.selectbox('Fuel Type', ('Petrol', 'Diesel', 'CNG'))
    transmission = st.sidebar.selectbox('Transmission', ('Manual', 'Automatic'))
    owner = st.sidebar.selectbox('Number of Owners', (0, 1, 2, 3))

    # Load the label encoder and scaler
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()

    # Preprocess input data
    input_data = preprocess_input(year, present_price, selling_type, km_driven, fuel_type, transmission, owner, label_encoder, scaler)

    # Make predictions
    prediction = model.predict(input_data)

    # Display the predicted selling price
    st.subheader('Predicted Selling Price:')
    st.write(f'${prediction[0]:,.2f}')

if __name__ == '__main__':
    main()
