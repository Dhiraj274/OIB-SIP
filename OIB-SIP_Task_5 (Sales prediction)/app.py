import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
# Load data
sales_data = pd.read_csv('Advertising.csv', index_col='Unnamed: 0')
# User input for TV, Radio, and Newspaper budgets
st.header('Enter Advertising Budgets:')
TV = st.number_input('TV Advertising Budget', value=0,step=None)
radio = st.number_input('Radio Advertising Budget', value=0,step=None)
newspaper = st.number_input('Newspaper Advertising Budget', value=0,step=None)
# User input dataframe
user_input = pd.DataFrame({'TV': [TV], 'Radio': [radio], 'Newspaper': [newspaper]})
# Show user input
st.subheader('User Input:')
st.write(user_input)
# Button to trigger prediction
if st.button('Predict Sales'):
    # Split the dataset
    X = sales_data[['TV', 'Radio', 'Newspaper']]
    y = sales_data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Make predictions
    prediction = model.predict(user_input)
    # Display prediction in a green box
    st.subheader('Sales Prediction:')
    st.success(f'The predicted sales value is: {prediction[0]:.2f}')