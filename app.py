import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------------
# Load the data
# ----------------------------------
@st.cache_data
def load_data():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
               'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
               'ca', 'thal', 'target']
    df = pd.read_csv('processed.cleveland.data', names=columns)
    
    # Handle missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.astype(float)
    
    # Convert target
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

df = load_data()

# Features and Target
X = df.drop('target', axis=1)
y = df['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------
# Train models
# ----------------------------------
log_reg = LogisticRegression(max_iter=3000)
log_reg.fit(X_train, y_train)

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgboost_model.fit(X_train, y_train)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("Heart Disease Prediction App ❤️")
st.markdown("""
This application predicts the likelihood of a patient having heart disease using different Machine Learning models:
- Logistic Regression
- Random Forest
- XGBoost

Upload patient data, select a model, and get instant predictions!
""")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', (0, 1)) # 0: female, 1: male
    cp = st.sidebar.slider('Chest Pain Type (0-3)', 0, 3, 1)
    trestbps = st.sidebar.slider('Resting Blood Pressure', 80, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    restecg = st.sidebar.slider('Resting ECG Results (0-2)', 0, 2, 1)
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 250, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', (0, 1))
    oldpeak = st.sidebar.slider('ST depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.slider('Slope of Peak Exercise ST Segment (0-2)', 0, 2, 1)
    ca = st.sidebar.slider('Number of Major Vessels (0-3)', 0, 3, 0)
    thal = st.sidebar.slider('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)', 0, 3, 2)
    
    data = {
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
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# ----------------------------------
# Model Selection
# ----------------------------------
model_choice = st.selectbox('Select the Machine Learning Model', ('Logistic Regression', 'Random Forest', 'XGBoost'))

# ----------------------------------
# Prediction
# ----------------------------------
if st.button('Predict'):
    if model_choice == 'Logistic Regression':
        prediction = log_reg.predict(input_df)
        prediction_proba = log_reg.predict_proba(input_df)
    elif model_choice == 'Random Forest':
        prediction = random_forest.predict(input_df)
        prediction_proba = random_forest.predict_proba(input_df)
    else:
        prediction = xgboost_model.predict(input_df)
        prediction_proba = xgboost_model.predict_proba(input_df)
    
    # result = 'Heart Disease Detected 💔' if prediction[0] == 1 else 'No Heart Disease ❤️'
    
    
# --- Display Predictions ---
    st.subheader('Prediction Result:')
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

# ----------------------------------
# Show Raw Data
# ----------------------------------
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Dataset')
    st.write(df)
