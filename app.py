import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("mood_data.csv")

# Show dataset preview in Streamlit
st.title("🧠 Student Mood Predictor")
st.write("### Preview of Student Mood Data")
st.dataframe(data)

# Define features and target
X = data[['SleepHours', 'StudyHours', 'ActivityLevel', 'CaffeineIntake',
          'SocialHours', 'DietQuality', 'ExerciseHours', 'ScreenTime']]
y = data['Mood']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
st.write(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# User input
st.write("### Enter Your Daily Routine")
sleep = st.slider("Hours of Sleep", 0, 10, 7)
study = st.slider("Hours of Study", 0, 12, 5)
activity = st.slider("Activity Level (1-10)", 1, 10, 5)
caffeine = st.slider("Caffeine Intake (cups)", 0, 10, 2)
social = st.slider("Hours of Socializing", 0, 10, 3)
diet = st.slider("Diet Quality (1-10)", 1, 10, 6)
exercise = st.slider("Exercise Hours", 0, 5, 1)
screen = st.slider("Screen Time (hours)", 0, 12, 6)

# Create input DataFrame
input_df = pd.DataFrame({
    'SleepHours': [sleep],
    'StudyHours': [study],
    'ActivityLevel': [activity],
    'CaffeineIntake': [caffeine],
    'SocialHours': [social],
    'DietQuality': [diet],
    'ExerciseHours': [exercise],
    'ScreenTime': [screen]
})

st.write("### Your Input Data")
st.dataframe(input_df)

# Prediction
prediction = model.predict(input_df)[0]

# Show prediction result
st.subheader("🎯 Predicted Mood:")
if prediction == "Happy":
    st.success("😊 You are in a **Happy Mood**! Keep up your routine.")
elif prediction == "Relaxed":
    st.info("😌 You are feeling **Relaxed**. Great balance in your lifestyle!")
elif prediction == "Depressed":
    st.error("😔 You seem **Depressed**. Take some rest and focus on self-care.")
elif prediction == "Stressed":
    st.warning("⚡ You are **Stressed**. Try relaxation techniques or reduce study load.")
else:
    st.write("🙂 You are in a **Neutral Mood**.")
