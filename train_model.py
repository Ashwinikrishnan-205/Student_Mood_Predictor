import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset with all features (80+ rows)
data = pd.read_csv("mood_data.csv")

# Features and target
X = data[['SleepHours','StudyHours','ActivityLevel','CaffeineIntake','SocialHours',
          'DietQuality','ExerciseHours','ScreenTime']]
y = data['Mood']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("mood_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
