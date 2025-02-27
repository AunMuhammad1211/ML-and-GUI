import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

@st.cache
def load_data():
    data = pd.read_csv("data.csv")
    return data

def preprocess_data(data):
    columns_name = [
        "Previous qualification", "Previous qualification (grade)", "Mother's occupation", 
        "Father's occupation", "Displaced", "Admission grade", "Scholarship holder", 
        "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", 
        "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)", 
        "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)", 
        "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)", 
        "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)", 
        "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)", 
        "Unemployment rate", "Inflation rate", "GDP", "Target"
    ]
    
    missing_columns = [col for col in columns_name if col not in data.columns]
    
    if missing_columns:
        st.error(f"The following required columns are missing: {missing_columns}")
        st.stop()
    
    data = data[columns_name]
    
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    X = data.drop(columns=['Target'])
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Target Distribution"])

    data = load_data()
    X_train, X_test, y_train, y_test, label_encoders = preprocess_data(data)
    model = train_model(X_train, y_train)

    if page == "Prediction":
        st.title("Student Dropout Prediction")
        
        input_data = {}
        for column in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[column]):
                input_data[column] = st.number_input(f"Enter value for {column}", value=X_train[column].mean())
            else:
                unique_values = X_train[column].unique()
                selected_value = st.selectbox(f"Select value for {column}", unique_values)
                input_data[column] = selected_value
        
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            
            prediction = model.predict(input_df)
            st.write(f"Prediction: {label_encoders['Target'].inverse_transform(prediction)[0]}")

    elif page == "Target Distribution":
        st.title("Target Variable Distribution")
        
        plt.figure(figsize=(5, 4))
        sns.countplot(x=data['Target'], palette='viridis')
        plt.title("Distribution of Target Variable")
        plt.xlabel("Category")
        plt.ylabel("Frequency")
        st.pyplot(plt)

if __name__ == "__main__":
    main()