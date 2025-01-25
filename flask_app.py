from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and vectorizer (same as before)
import pickle
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
    data = request.json
    news_text = data.get('text', '')

    if not news_text.strip():
        return jsonify({"error": "No text provided"}), 400

    text_tfidf = vectorizer.transform([news_text])
    prediction = model.predict(text_tfidf)[0]
    confidence = model.predict_proba(text_tfidf).max() * 100

    return jsonify({
        "label": "TRUE" if prediction == 1 else "FALSE",
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
