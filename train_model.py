import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

print("test 1")

# Load fake and real datasets (no headers)
fake_data = pd.read_csv('data/dataset_fake.csv', header=None, on_bad_lines='skip')
real_data = pd.read_csv('data/dataset_true.csv', header=None, on_bad_lines='skip')

print("test 2")

# Clean and preprocess text
fake_data[0] = fake_data[0].astype(str).apply(lambda x: x.split(',', 1)[1].strip().strip('"') if ',' in x else x)
real_data[0] = real_data[0].astype(str).apply(lambda x: x.split(',', 1)[1].strip().strip('"') if ',' in x else x)

print("test 3")

# Add labels: 0 for fake news, 1 for real news
fake_data['label'] = 0
real_data['label'] = 1

print("test 4")

# Combine datasets
data = pd.concat([fake_data, real_data], ignore_index=True)

print("test 5")

# Features and labels
X = data[0]
y = data['label']

print("test 6")

# TF-IDF vectorization with optimizations for performance
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

print("test 7")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.05, random_state=42)

print("test 8")

# Random Forest model training with optimizations
model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, verbose=1)
model.fit(X_train, y_train)

print("test 9")

# Predictions
prediction_test = model.predict(X_test)

print("test 10")

# Save the model and vectorizer
with open('models/fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

# Print accuracy
print("Accuracy =", metrics.accuracy_score(y_test, prediction_test))

print("test 11")
