import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv("mood_data.csv")

# Features and Target
X = data[['SleepHours', 'StudyHours', 'ActivityLevel']]
y = data['Mood']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "mood_model.pkl")

print("Model trained and saved successfully!")
