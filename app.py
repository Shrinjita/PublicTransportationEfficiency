import streamlit as st
import pandas as pd
from train_model import make_prediction  # Adjust this according to your prediction function
from eda import create_visualizations  # Adjust to your EDA function name

# Streamlit app header
st.title("Public Transportation Emissions Prediction")

# Input data for prediction
input_data = st.text_input("Enter data for prediction:")

if st.button("Predict"):
    prediction = make_prediction(input_data)
    st.write(f"Prediction Result: {prediction}")

# Load and display visualizations
visualizations = create_visualizations()
for fig in visualizations:
    st.plotly_chart(fig)
