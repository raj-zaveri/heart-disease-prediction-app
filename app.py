import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Heart Disease Prediction App", layout="wide")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('processed.cleveland.data', names=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ])
    data.replace('?', np.nan, inplace=True)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data.dropna(inplace=True)
    data['target'] = data['target'].apply(lambda x: 1 if int(x) > 0 else 0)
    return data

df = load_data()

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Heart Disease Prediction"])

# Input form
if page == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction App")
    st.markdown("---")

    st.header("Enter Patient Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Age', 20, 100, 50)
        trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
        chol = st.slider('Serum Cholestoral (mg/dl)', 100, 600, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)', [0, 1])

    with col2:
        sex = st.selectbox('Sex (0 : Female; 1 : Male )', [0, 1])
        cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
        restecg = st.selectbox('Resting ECG (0-2)', [0, 1, 2])
        thalach = st.slider('Max Heart Rate Achieved', 60, 250, 150)

    with col3:
        exang = st.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
        oldpeak = st.number_input('Oldpeak (ST depression)', 0.0, 6.0, step=0.1)
        slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels Colored (0-3)', [0, 1, 2, 3])
        thal = st.selectbox('Thalassemia (1 = Normal; 2 = Fixed defect; 3 = Reversable defect)', [1, 2, 3])

    model_choice = st.selectbox('Select Model', ['Logistic Regression', 'Random Forest', 'XGBoost'])

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)

    if st.button('Predict'):
        if model_choice == 'Logistic Regression':
            prediction = logistic_model.predict(input_scaled)[0]
            prediction_proba = logistic_model.predict_proba(input_scaled)[0][1]
        elif model_choice == 'Random Forest':
            prediction = rf_model.predict(input_scaled)[0]
            prediction_proba = rf_model.predict_proba(input_scaled)[0][1]
        else:
            prediction = xgb_model.predict(input_scaled)[0]
            prediction_proba = xgb_model.predict_proba(input_scaled)[0][1]

        # Display result
        if prediction == 0:
            st.success('✅ No Heart Disease Detected.')
        else:
            st.error('⚠️ High Risk of Heart Disease Detected!')

        st.markdown("---")
        st.subheader("🔎 Risk Probability")
        st.write(f"**Probability of Heart Disease:** {prediction_proba * 100:.0f}%")


        # 🎯 Visualization
        st.subheader("🔎 Prediction Result Visualization")

        # Risk Gauge Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': 'lightgreen'},
                    {'range': [40, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction_proba * 100
                }
            }
        ))

        st.plotly_chart(fig)


        # -----------------------
        # Footer
        # -----------------------
        st.markdown("---")
        st.markdown("Created by Raj Zaveri 🚀| Powered by Machine Learning")
