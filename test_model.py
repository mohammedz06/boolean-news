import pickle

# Load the trained model
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Function to predict real or fake news with confidence
def predict_news(news_text):
    # Transform the input text using the loaded vectorizer
    text_tfidf = tfidf.transform([news_text])
    # Get prediction probabilities
    prediction_prob = model.predict_proba(text_tfidf)
    # Get the predicted class (0 for fake, 1 for real)
    prediction = model.predict(text_tfidf)
    
    # Probability of the prediction
    prob = prediction_prob[0][prediction[0]] * 100  # Convert to percentage
    return ("Fake News", prob) if prediction[0] == 0 else ("Real News", prob)

# Example news to test
news_to_test = input("Enter a news article: ")
result, confidence = predict_news(news_to_test)
print(f"The news is: {result} with {confidence:.2f}% confidence.")
