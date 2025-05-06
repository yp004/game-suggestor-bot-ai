import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
df = pd.read_csv('data/training_data.csv')

# Features and labels
X = df['message']
y = df[['genre', 'platform', 'mood']]

# Text vectorization
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Multi-label classification
model = MultiOutputClassifier(RandomForestClassifier())
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump((model, vectorizer), 'model/game_classifier.pkl')

print("Model trained and saved successfully!")
