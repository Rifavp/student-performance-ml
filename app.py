import streamlit as st
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset.csv")

X = data[["Study_Hours", "Attendance", "Previous_Mark"]]

y = data["Result"]

decision_tree = DecisionTreeClassifier(random_state=42)

decision_tree.fit(X, y)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X, y)
st.title("Student Performance Prediction")
st.write("Decision Tree vs Random Forest")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_mark = st.number_input(
    "Previous Exam Mark",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

if st.button("Predict"):
    
    student = [[
        study_hours,
        attendance,
        previous_mark
    ]]

    dt_prediction = decision_tree.predict(student)[0]

    rf_prediction = random_forest.predict(student)[0]

    st.subheader("Prediction Results")

    st.write("Decision Tree:", dt_prediction)

    st.write("Random Forest:", rf_prediction)