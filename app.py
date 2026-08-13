import streamlit as st
import pandas as pd

st.title("Customer Churn Prediction")

st.write("Predict whether a customer will leave the company.")

data = pd.read_csv("customer_churn.csv")

st.subheader("Customer Data")
st.dataframe(data)

st.subheader("Prediction")

if st.button("Predict Churn"):
    st.success("Prediction completed!")
