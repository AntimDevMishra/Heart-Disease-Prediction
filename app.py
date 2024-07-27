import streamlit as st
import pandas as pd
import pickle


st.markdown(
"""
    # :rainbow[Heart Disease Prediction App]
""")

st.markdown("""
            #### _This app predicts if a patient has heart disease_
            ###### _Data is recorded from Realtime Sensors_
            """)

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    Sex = st.sidebar.selectbox('Sex (0: Female, 1: Male)', (0, 1))
    Age = st.sidebar.number_input('Age', min_value=29, max_value=77, value=54)
    Body_Temp = st.sidebar.number_input('Body Temp (°C)', min_value=35.0, max_value=42.0, value=37.0)
    Systolic_BP = st.sidebar.number_input('Systolic BP', min_value=90, max_value=200, value=130)
    Diastolic_BP = st.sidebar.number_input('Diastolic BP', min_value=60, max_value=120, value=80)
    Pulse_Rate = st.sidebar.number_input('Pulse Rate', min_value=40, max_value=180, value=70)
    Oxygen_Saturation = st.sidebar.number_input('Oxygen Saturation', min_value=85, max_value=100, value=98)
    Height = st.sidebar.number_input('Height (cm)', min_value=140, max_value=200, value=170)
    Weight = st.sidebar.number_input('Weight (kg)', min_value=40, max_value=150, value=70)
    BMI = st.sidebar.number_input('BMI', min_value=15.0, max_value=40.0, value=24.0)
    Cholesterol_Level = st.sidebar.selectbox('Cholesterol Level (1: Low, 2: Normal, 3: High)', (1, 2, 3))
    Smoking_Status = st.sidebar.selectbox('Smoking Status (0: Non-smoker, 1: Smoker)', (0, 1))

    data = {
        'Sex': Sex,
        'Age': Age,
        'Body Temp (°C)': Body_Temp,
        'Systolic BP': Systolic_BP,
        'Diastolic BP': Diastolic_BP,
        'Pulse Rate': Pulse_Rate,
        'Oxygen Saturation': Oxygen_Saturation,
        'Height (cm)': Height,
        'Weight (kg)': Weight,
        'BMI': BMI,
        'Cholesterol Level': Cholesterol_Level,
        'Smoking Status': Smoking_Status
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encoding of categorical features
input_df = pd.get_dummies(input_df, columns=['Sex', 'Cholesterol Level', 'Smoking Status'])

# Ensuring all dummy variables exist in the input data
expected_columns = ['Sex_0', 'Sex_1', 'Cholesterol Level_1', 'Cholesterol Level_2', 'Cholesterol Level_3',
                    'Smoking Status_0', 'Smoking Status_1']
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match the training data
input_df = input_df[['Age', 'Body Temp (°C)', 'Systolic BP', 'Diastolic BP', 'Pulse Rate', 'Oxygen Saturation', 
                     'Height (cm)', 'Weight (kg)', 'BMI', 'Sex_0', 'Sex_1', 'Cholesterol Level_1', 
                     'Cholesterol Level_2', 'Cholesterol Level_3', 'Smoking Status_0', 'Smoking Status_1']]

st.subheader('User Input features')
st.write(input_df)

# Load the trained Random Forest model
load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Button for prediction
if st.button('Predict'):
    # Apply model to make predictions
    prediction = load_clf.predict(input_df)
    prediction_proba = load_clf.predict_proba(input_df)

    st.markdown('## :rainbow[Prediction]')
    st.write('Heart Disease ✔️' if prediction[0] else 'No Heart Disease ❌')

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
