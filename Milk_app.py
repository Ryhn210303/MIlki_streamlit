import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title of the app
st.title("Milk Quality Prediction")

# Manual Input Form for Data
st.write("### Input Milk Quality Data Manually")

def user_input_features():
    taste = st.selectbox("Taste (Good: 1, Bad: 0)", [0, 1])
    odor = st.selectbox("Odor (Good: 1, Bad: 0)", [0, 1])
    ph = st.number_input("pH (e.g., 6.7)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    temperature = st.number_input("Temperature (°C, e.g., 25)", min_value=-50.0, max_value=100.0, value=25.0, step=0.1)
    colour = st.number_input("Colour (e.g., 1.5)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)

    data = {
        'Taste': taste,
        'Odor': odor,
        'pH': ph,
        'Temprature': temperature,
        'Colour': colour
    }
    return pd.DataFrame([data])

input_data = user_input_features()
st.write("### Input Data:")
st.write(input_data)

# Upload CSV file for training
uploaded_file = st.file_uploader("Upload your CSV file for training", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.write(data.head())

    # Data Preprocessing
    st.write("### Data Preprocessing:")
    st.write("Checking for null values...")
    st.write(data.isnull().sum())

    # Encoding categorical features
    encoder = LabelEncoder()
    data['Taste'] = encoder.fit_transform(data['Taste'])
    data['Odor'] = encoder.fit_transform(data['Odor'])
    data['Grade'] = data['Grade'].map({'low': 0, 'medium': 1, 'high': 2})

    # Standardizing numerical features
    scaler = StandardScaler()
    data[['pH', 'Temprature', 'Colour']] = scaler.fit_transform(data[['pH', 'Temprature', 'Colour']])

    # Splitting data
    X = data.drop('Grade', axis=1)
    y = data['Grade']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    # Predict the quality of the manually input data
    input_data_scaled = input_data.copy()
    input_data_scaled[['pH', 'Temprature', 'Colour']] = scaler.transform(input_data_scaled[['pH', 'Temprature', 'Colour']])

    pred_dt = dt_model.predict(input_data_scaled)
    pred_label_dt = "high" if pred_dt[0] == 2 else "low"

    pred_nb = nb_model.predict(input_data_scaled)
    pred_label_nb = "high" if pred_nb[0] == 2 else "low"

    # Display prediction results
    st.write("### Prediction Results:")
    st.write(f"**Decision Tree Prediction:** {pred_label_dt}")
    st.write(f"**Naive Bayes Prediction:** {pred_label_nb}")