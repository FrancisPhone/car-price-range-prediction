import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu


# Load your trained model
with open('car_price_prediction.model', 'rb') as model_file:
    old_model = pickle.load(model_file)

with open('model.pkl', 'rb') as model_file:
    new_model = pickle.load(model_file)


# Scaling model
with open('scaler.pickle', 'rb') as handle:
    scaler = pickle.load(handle)


def old():
    st.title("Car Price Prediction")

    # Inputs
    year = st.number_input('Year', min_value=1900, max_value=2024, value=2020)
    transmission = st.selectbox('Transmission', ['Auto', 'Manual'])
    engine = st.number_input('Engine (CC)', min_value=624, value=1462)
    max_power = st.number_input('Max Power (HP)', min_value=68, value=90)

    # Encode transmission
    transmission_encoded = 1 if transmission == 'Auto' else 0

    # Predict button
    if st.button('Predict'):
        input_data = np.array([[year, transmission_encoded, engine, max_power]])
        prediction = old_model.predict(input_data)
        price = np.exp(prediction[0])
        st.write(f'The predicted selling price is ${price:.2f}')

    st.text_area("This is a Random Forest Regressor model to predict the selling price of a car according to the following inputs :)",
    """ **Instructions for using the Car Price Prediction App:**

        1. **Year:** Enter the year of the car. Use the number input field to select a value between 1900 and 2024.
        2. **Transmission:** Select the type of transmission (Auto or Manual) from the dropdown menu.
        3. **Engine (CC):** Enter the engine capacity in cubic centimeters. Use the number input field to select a value.
        4. **Max Power (HP):** Enter the maximum power in horsepower. Use the number input field to select a value.

        After entering the values, click the "Predict" button to see the estimated selling price of the car.""", height=300)


def new():
    st.title("Car Price Prediction (New)")

    # Inputs
    year = st.number_input('Year', min_value=1900, max_value=2024, value=2020)
    transmission = st.selectbox('Transmission', ['Auto', 'Manual'])
    engine = st.number_input('Engine (CC)', min_value=624, value=1462)
    max_power = st.number_input('Max Power (HP)', min_value=68, value=90)

    # Encode transmission
    transmission_encoded = 1 if transmission == 'Auto' else 0

    # Predict button
    if st.button('Predict'):
        input_data = np.array([[year, transmission_encoded, engine, max_power]])
        data = scaler.transform(input_data)
        intercept = np.ones((data.shape[0], 1))
        data = np.concatenate((intercept, data), axis=1)
        prediction = new_model.predict(data)
        price = np.exp(prediction[0])
        st.write(f'The predicted selling price is ${price:.2f}')

    st.text_area("This is a Linear Regression model which surprisingly increases in accuracy "
                 "due to the Xavier weight initialization and Momentum gradient descent.",
    """ **Instructions for using the Car Price Prediction App:**

        1. **Year:** Enter the year of the car. Use the number input field to select a value between 1900 and 2024.
        2. **Transmission:** Select the type of transmission (Auto or Manual) from the dropdown menu.
        3. **Engine (CC):** Enter the engine capacity in cubic centimeters. Use the number input field to select a value.
        4. **Max Power (HP):** Enter the maximum power in horsepower. Use the number input field to select a value.

        After entering the values, click the "Predict" button to see the estimated selling price of the car.""", height=300)


with st.sidebar:
    selected = option_menu(
        "Models",
        ["Old", "New"],
        icons=["bi bi-android", "bi bi-robot"],  # Optional icons
        menu_icon="bi bi-cpu",  # Optional menu icon
        default_index=0,  # Default selected tab
    )

# Display the selected page
if selected == "Old":
    old()
elif selected == "New":
    new()






