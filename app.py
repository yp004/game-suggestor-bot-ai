from flask import Flask, render_template, request, jsonify
import joblib
import json
import re

app = Flask(__name__)

# Load model and vectorizer
model, vectorizer = joblib.load('model/game_classifier.pkl')

# Load games database
with open('games.json', 'r') as f:
    games_db = json.load(f)

# Greeting & casual talk detection
greetings = ["hi", "hello", "hey", "how are you", "what's up", "yo", "sup", "good morning", "good evening"]

# Smart fallback flag
last_was_nomatch = False

# Predict intent and preferences
def detect_intent(user_msg):
    message = user_msg.lower()
    if any(greet in message for greet in greetings):
        return "greeting"
    elif re.search(r"\b(game|suggest|recommend|like|play|similar)\b", message):
        return "game_request"
    else:
        return "chat"

def predict_preferences(user_message):
    X_vec = vectorizer.transform([user_message])
    prediction = model.predict(X_vec)
    genre, platform, mood = prediction[0]
    return genre, platform, mood

def suggest_games(genre, platform, mood):
    suggestions = []
    for game in games_db:
        if genre in game['genre'] and platform in game['platforms'] and mood in game['mood']:
            suggestions.append(game['name'])
    return suggestions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot_response():
    global last_was_nomatch
    user_msg = request.json['message']
    intent = detect_intent(user_msg)

    if intent == "greeting":
        reply = "Hey gamer! 🎮 What kind of game are you in the mood for today?"
        last_was_nomatch = False
    elif intent == "chat":
        reply = "Hmm, sounds cool. Wanna talk games? 👾"
        last_was_nomatch = False
    elif intent == "game_request":
        genre, platform, mood = predict_preferences(user_msg)
        suggestions = suggest_games(genre, platform, mood)
        if suggestions:
            reply = f"You might enjoy: {', '.join(suggestions)}"
            last_was_nomatch = False
        elif not last_was_nomatch:
            reply = "Hmm, I didn’t find a perfect match. Try describing it differently or be more specific! 🎯"
            last_was_nomatch = True
        else:
            reply = "Still no match, but I’m trying! 😅"
    else:
        reply = "Hmm, I didn’t get that. Want a game suggestion? 🎮"

    return jsonify({'reply': reply})
if __name__ == "__main__":
    app.run(debug=True)

