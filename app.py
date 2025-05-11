import streamlit as st
import pandas as pd
from joblib import load
import dill

# Load the pretrained model
with open('emp_churn_pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('emp_churn_feature_dict.pkl')

# Function to predict churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.markdown("""
    <div style='background-color: #0e1117; padding: 20px; border-radius: 10px;'>
        <h1 style='color: #00ffcc; text-align: center;'>🔮 Customer Churn Prediction</h1>
        <h4 style='color: #cccccc; text-align: center;'>Developed by: <b>Muhammad Usama Alam</b></h4>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("---")

# Display categorical features
st.markdown('### 🧩 Categorical Features')
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals={}
with st.expander("Click to enter categorical values", expanded=True):
    for i, col in enumerate(categorical_input.get('Column Name').values()):
        categorical_input_vals[col] = st.selectbox(col, categorical_input.get('Members')[i],key=col)

# Display numerical features
st.markdown("### 📊 Numerical Features")
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals={}
for col in  numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.number_input(col,key=col)

# Combine numerical and categorical input dicts
input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))

input_data= pd.DataFrame.from_dict(input_data,orient='index').T

st.markdown("---")


# Churn Prediction
if st.button('🚀 Predict'):
    with st.spinner("Predicting..."):
        prediction = predict_churn(input_data)[0]
        translation_dict = {"Stay": "Not Expected To Leave", "Leave": "Expected To Leave"}
        prediction_translate = translation_dict.get(prediction)
        #st.write(f'The Prediction is **{prediction}**, Hence customer is **{prediction_translate}** to churn.')    

        if prediction == "Stay":
            st.success(f'🎯 Prediction: **{prediction}** — Customer is **{prediction_translate}** to churn.')
            st.balloons()
        else:
            st.info(f'✅ Prediction: **{prediction}** — Customer is **{prediction_translate}** to churn.')

st.markdown("""
    <br><hr>
    <div style='text-align: center; color: gray;'>
        Developed by Muhammad Usama Alam. All rights reserved.
    </div>
""", unsafe_allow_html=True)            