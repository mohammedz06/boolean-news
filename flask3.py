from flask import Flask, request, jsonify
import sqlite3
import pickle
import sentimentFunc  # Ensure the bias_check function is in this module

app = Flask(__name__)

# Load the model and vectorizer
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Root route
@app.route('/')
def home():
    return "Boolean-News"

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts whether a news article is fake or real.
    """
    data = request.json
    news_text = data.get('text', '')

    if not news_text.strip():
        return jsonify({"error": "No text provided"}), 400

    text_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf).max() * 100

    # Bias analysis (ensure that bias_check is imported and functional)
    polarity, subjectivity = sentimentFunc.bias_check(news_text)

    return jsonify({
        "label": "TRUE" if prediction == 1 else "FALSE",
        "confidence": round(confidence, 2),
        "polarity": round(polarity, 2),
        "subjectivity": round(subjectivity, 2)
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

    # Validate input
    if not news_text or not predicted_label or not actual_label:
        return jsonify({"error": "Invalid feedback data. Please provide 'text', 'predicted_label', and 'actual_label'."}), 400

    # Save feedback in the feedback.db database
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
