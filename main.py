from flask import Flask, request, jsonify, render_template
import sqlite3
import pickle
from textblob import TextBlob

app = Flask(__name__)

# Load the model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# Bias analysis function
def bias_check(news):
    """
    Analyzes the polarity and subjectivity of the text using TextBlob.
    """
    blob = TextBlob(news)
    sentiment = blob.sentiment

    # Polarity: -1 to 1 (negative to positive sentiment)
    polarity = sentiment.polarity

    # Subjectivity: 0 (completely objective) to 1 (completely subjective)
    subjectivity = sentiment.subjectivity

    return polarity, subjectivity


# Root route
@app.route('/')
def home():
    return render_template("main.html")


# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    confidence = None
    polarity = None
    subjectivity = None
    news_text = None

    if request.method == 'POST':
        if request.is_json:  # Handle JSON input (API)
            data = request.json
            news_text = data.get('text', '').strip()
        else:  # Handle form input (HTML form submission)
            news_text = request.form.get('news_text', '').strip()

        if news_text:
            # Pass the text to the model
            text_tfidf = vectorizer.transform([news_text])
            pred = model.predict(text_tfidf)[0]
            confidence = model.predict_proba(text_tfidf).max() * 100
            prediction = "TRUE" if pred == 1 else "FALSE"

            # Perform bias analysis
            polarity, subjectivity = bias_check(news_text)
            polarity = round((polarity + 1) * 50, 2)  # Convert -1 to 1 range into 0-100%
            subjectivity = round(subjectivity * 100, 2)  # Convert 0-1 to percentage
        else:
            return render_template("main.html", error="No text provided. Please enter an article.")

    # If it's a form submission, render the HTML template with the results
    if not request.is_json:
        return render_template(
            "main.html",
            prediction=prediction,
            confidence=round(confidence, 2) if confidence else None,
            polarity=polarity if polarity else None,
            subjectivity=subjectivity if subjectivity else None,
            news_text=news_text,
        )

    # If it's a JSON API request, return the prediction and confidence
    return jsonify({
        "label": prediction,
        "confidence": round(confidence, 2) if confidence else None,
        "polarity": polarity,
        "subjectivity": subjectivity,
    })


# Feedback route
@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Collects user feedback on predictions and stores it in the feedback.db database.
    """
    data = request.json

    # Extract feedback data
    news_text = data.get('text', '').strip()
    predicted_label = data.get('predicted_label', '').strip()
    actual_label = data.get('actual_label', '').strip()

    if not news_text or not predicted_label or not actual_label:
        return jsonify({"error": "Invalid feedback data"}), 400

    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO feedback (text, predicted_label, actual_label)
            VALUES (?, ?, ?)
        ''', (news_text, predicted_label, actual_label))
        conn.commit()
        conn.close()
        return jsonify({"message": "Feedback received!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
