import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("mood_model.pkl")

st.title("🧠 Student Mood Predictor")
st.write("Predict your mood based on daily habits and activities.")

# --- Load CSV dataset ---
mood_data = pd.read_csv("mood_data.csv")
st.subheader("Sample Input Data")
st.dataframe(mood_data)

# --- User Inputs ---
sleep = st.number_input("Hours of Sleep", 0, 24, 6)
study = st.number_input("Hours of Study", 0, 24, 5)
activity = st.number_input("Activity Level (1-10)", 1, 10, 5)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
exercise = st.number_input("Exercise Hours", 0, 5, 1)
social = st.slider("Social Interaction (1-10)", 1, 10, 5)

# --- Show current input as a table ---
input_display_df = pd.DataFrame(
    [[sleep, study, activity, stress, exercise, social]],
    columns=['Sleep Hours', 'Study Hours', 'Activity Level', 'Stress Level', 'Exercise Hours', 'Social Interaction']
)
st.subheader("Your Current Input Data")
st.dataframe(input_display_df)

# --- Prepare input for model (original features used in training) ---
input_df = pd.DataFrame([[sleep, study, activity]],
                        columns=['SleepHours','StudyHours','ActivityLevel'])

# --- Predict Mood ---
if st.button("Predict Mood"):
    prediction = model.predict(input_df)
    mood_label = ["Happy","Stressed","Depressed"][prediction[0]]
    st.success(f"📝 The predicted mood is: {mood_label}")

    # --- Dynamic advice based on mood ---
    if mood_label == "Happy":
        st.info("Keep up the good mood! 😄 Stay productive and healthy!")
    elif mood_label == "Stressed":
        st.info("Take short breaks, relax, and do some deep breathing. 🧘‍♂️")
    else:
        st.info("Consider talking to someone, take rest, and reduce stress. 💛")
