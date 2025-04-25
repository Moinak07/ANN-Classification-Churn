import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="centered", initial_sidebar_state="expanded")

st.title('Customer Churn Prediction')
st.markdown('---')
st.header('Enter Customer Information')

# User input
geography = st.selectbox('Select Geography', onehot_encoder_geo.categories_[0], key="geo")
gender = st.selectbox('Select Gender', label_encoder_gender.classes_, key="gender")
age = st.slider('Age', 18, 92, key="age")
balance = st.number_input('Account Balance ($)', 0.0, 1_000_000.0, step=100.0, key="balance")
credit_score = st.number_input('Credit Score', 300, 850, step=1, key="credit_score")
estimated_salary = st.number_input('Estimated Salary ($)', 20_000.0, 500_000.0, step=1000.0, key="salary")
tenure = st.slider('Tenure (Years)', 0, 10, key="tenure")
num_of_products = st.slider('Number of Products', 1, 4, key="num_products")
has_cr_card = st.selectbox('Has Credit Card?', ['No', 'Yes'], key="has_cr_card")
is_active_member = st.selectbox('Is Active Member?', ['No', 'Yes'], key="is_active_member")

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display results
st.markdown('---')
st.subheader('Prediction Result')
st.write(f'**Churn Probability:** {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.markdown('<h3 style="color: red;">The customer is likely to churn.</h3>', unsafe_allow_html=True)
else:
    st.markdown('<h3 style="color: green;">The customer is not likely to churn.</h3>', unsafe_allow_html=True)

# Additional info section
st.markdown('---')
st.subheader('Input Details:')
st.write(f'**Geography:** {geography}')
st.write(f'**Gender:** {gender}')
st.write(f'**Age:** {age}')
st.write(f'**Account Balance:** ${balance:,.2f}')
st.write(f'**Credit Score:** {credit_score}')
st.write(f'**Estimated Salary:** ${estimated_salary:,.2f}')
st.write(f'**Tenure:** {tenure} years')
st.write(f'**Number of Products:** {num_of_products}')
st.write(f'**Has Credit Card:** {has_cr_card}')
st.write(f'**Active Member:** {is_active_member}')
