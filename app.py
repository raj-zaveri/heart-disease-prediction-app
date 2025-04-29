import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit page configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="wide", page_icon="💓")

# Title section
st.markdown("""
    <h1 style='text-align: center; color: crimson;'>💓 Heart Disease Prediction App</h1>
    <p style='text-align: center; font-size:18px;'>Input patient details to predict the likelihood of heart disease using ML models</p>
""", unsafe_allow_html=True)

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

# Sidebar input fields
st.sidebar.title("📝 Patient Info")
st.sidebar.markdown("Fill in the patient's details:")

def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dL)", 120, 600, 245)
    fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.radio("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope", [0, 1, 2])
    ca = st.sidebar.selectbox("Major Vessels Colored", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    features = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame([features])

input_df = user_input_features()

# Model selection
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Select Classifier")
model_name = st.sidebar.radio("Choose an ML model", 
                              ["Logistic Regression", "Random Forest", "XGBoost"])

# Data preparation
X = df.drop("target", axis=1)
y = df["target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Apply same scaling to user input
input_scaled = scaler.transform(input_df)

# Model training
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
prediction = model.predict(input_scaled)[0]
prediction_proba = model.predict_proba(input_scaled)[0][1]

# Layout with columns
col1, col2 = st.columns([1, 2])

# Results
with col1:
    st.subheader("🔍 Prediction Results")
    st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    st.metric("Patient Risk", "💔 At Risk" if prediction == 1 else "✅ No Risk")
    st.progress(prediction_proba)

with col2:
    with st.expander("📊 Detailed Evaluation"):
        st.text("Classification Report:")
        st.code(classification_report(y_test, y_pred), language='text')
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

# Footer
st.markdown("""
<hr>
<p style='text-align: center;'>
Made with ❤️ using Streamlit | <a href='https://github.com/raj-zaveri'>GitHub</a>
</p>
""", unsafe_allow_html=True)
