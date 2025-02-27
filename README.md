# Student Dropout Prediction App

## Overview
This is a **Streamlit-based web application** that predicts student dropout probabilities using a **Random Forest Classifier**. It allows users to input student-related data and get predictions on whether the student is likely to **graduate, drop out, or remain enrolled**.

## Features
- **Predicts Student Dropout** based on various factors.
- **Displays Target Distribution** using Seaborn visualizations.
- **Handles Data Preprocessing** (Label Encoding, Missing Columns Check).
- **Trains a Random Forest Model** for classification.
- **User-Friendly Input Interface** for making predictions.

## Installation and Working

Clone the repository

```bash
  git clone https://github.com/AunMuhammad1211/ML-and-GUI
```

To run this application, install the required dependencies:
```bash
pip install streamlit pandas matplotlib seaborn scikit-learn
```

How to run the application
```bash
streamlit run app.py
```
## How It Works
Load Dataset: The app reads data.csv and ensures all required columns are present.
Preprocess Data: Converts categorical features into numerical using Label Encoding.
Train Model: Uses a Random Forest Classifier to learn from student data.
Make Predictions: Users can enter student details via the UI to predict outcomes.
Visualize Target Distribution: The app displays a bar chart of student status categories.

## App Interface
This app comprises of two sections:

Prediction Page: Enter student data and click "Predict" to see the result.
![image](https://github.com/user-attachments/assets/1ded9ade-2951-4ac0-a87b-0c59b9b39916)

Target Distribution Page: View the distribution of target labels using a bar chart.
![image](https://github.com/user-attachments/assets/38d691c5-dfc1-4466-8146-3edcb2f4fde6)











