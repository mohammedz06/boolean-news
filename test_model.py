import pickle

# Load the trained model
with open('models/fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Function to predict real or fake news
def predict_news(news_text):
    # Transform the input text using the loaded vectorizer
    text_tfidf = tfidf.transform([news_text])
    # Make a prediction
    prediction = model.predict(text_tfidf)
    return "Fake News" if prediction[0] == 0 else "Real News"

# Example news to test
news_to_test = input("Enter a news article: ")
result = predict_news(news_to_test)
print(f"The news is: {result}")
