import streamlit as st
import pandas as pd
import pickle

# Load model
with open("mood_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🧠 Student Mood Predictor")

# Load dataset to display
data = pd.read_csv("mood_data.csv")
st.subheader("📊 Input Data Preview")
st.dataframe(data)

st.subheader("Enter Your Details")

# Input features
SleepHours = st.number_input("Hours of Sleep", min_value=0, max_value=24, value=7)
StudyHours = st.number_input("Hours of Study", min_value=0, max_value=24, value=5)
ActivityLevel = st.slider("Activity Level (1-10)", 1, 10, 5)
CaffeineIntake = st.slider("Caffeine Intake (1-10)", 0, 10, 2)
SocialHours = st.number_input("Social Hours", min_value=0, max_value=24, value=3)
DietQuality = st.slider("Diet Quality (1-10)", 1, 10, 7)
ExerciseHours = st.number_input("Exercise Hours", min_value=0, max_value=24, value=2)
ScreenTime = st.number_input("Screen Time Hours", min_value=0, max_value=24, value=5)

# Create DataFrame for prediction
input_df = pd.DataFrame([[SleepHours, StudyHours, ActivityLevel, CaffeineIntake,
                          SocialHours, DietQuality, ExerciseHours, ScreenTime]],
                        columns=['SleepHours','StudyHours','ActivityLevel','CaffeineIntake',
                                 'SocialHours','DietQuality','ExerciseHours','ScreenTime'])

st.subheader("Your Input Data")
st.dataframe(input_df)

# Prediction
if st.button("Predict Mood"):
    prediction = model.predict(input_df)[0]

    # Custom message based on mood
    message = ""
    if prediction == "Happy":
        message = "😊 You are feeling good! Keep it up!"
    elif prediction == "Stressed":
        message = "😟 You seem stressed. Take short breaks!"
    elif prediction == "Depressed":
        message = "😢 Mood is low. Take rest and relax!"
    elif prediction == "Relaxed":
        message = "😌 You are relaxed. Great job!"
    elif prediction == "Neutral":
        message = "😐 You are neutral. Maintain balance!"

    st.subheader("Predicted Mood")
    st.write(f"**{prediction}**")
    st.write(message)
