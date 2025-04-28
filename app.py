import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- Set Page Configuration ---
st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="wide")

# --- Title and Description ---
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
This application predicts the likelihood of a patient having heart disease using different Machine Learning models:
- Logistic Regression
- Random Forest
- XGBoost

Upload patient data, select a model, and get instant predictions!
""")

# --- Load Dataset ---
@st.cache_data
def load_data():
    data = pd.read_csv('processed.cleveland.data', names=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
        'ca', 'thal', 'target'
    ])
    data.replace('?', np.nan, inplace=True)
    data['target'] = data['target'].apply(lambda x: 1 if int(x) > 0 else 0)
    return data

df = load_data()

# --- Sidebar for User Input ---
st.sidebar.header('User Input Options')
model_choice = st.sidebar.selectbox("Select the Machine Learning Model", 
                                    ("Logistic Regression", "Random Forest", "XGBoost"))

st.sidebar.write("")

st.sidebar.subheader('Input Patient Details')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex (0 = female, 1 = male)', (0, 1))
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)', (0, 1))
    restecg = st.sidebar.slider('Resting ECG (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0)
    slope = st.sidebar.slider('Slope of Peak Exercise ST Segment (0-2)', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', 1, 3, 2)

    features = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# --- Split Data for Training ---
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Models ---
if model_choice == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000)
elif model_choice == 'Random Forest':
    model = RandomForestClassifier()
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)

# --- Display Predictions ---
st.subheader("Patient Data Input")
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction Result:")
if prediction[0] == 0:
    st.success('✅ No Heart Disease Detected')
else:
    st.error('⚠️ High Risk of Heart Disease')

st.subheader("Prediction Probability:")
st.write(f"Chances of No Heart Disease: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Chances of Heart Disease: {prediction_proba[0][1]*100:.2f}%")

# --- Footer ---
st.markdown("""---""")
st.caption("Created by Raj Zaveri 🚀 | Powered by Machine Learning")
