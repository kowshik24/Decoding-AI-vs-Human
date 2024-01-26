from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load trained model
with open('model_svc.pkl', 'rb') as f:
    model_svc = pickle.load(f)

# Load TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Load StandardScaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    text = request.form.get('text')

    # Perform the same feature engineering as during training
    word_count = len(text.split())
    avg_word_length = np.mean([len(word) for word in text.split()])

    # Transform the text data using the loaded TfidfVectorizer
    X_tfidf = tfidf_vectorizer.transform([text]).toarray()

    # Combine the TF-IDF features with the additional features
    additional_features = np.array([word_count, avg_word_length]).reshape(1, -1)
    X_combined = np.hstack([X_tfidf, additional_features])

    # Combine the TF-IDF features with the additional features
    #X_combined = np.hstack([X_tfidf, np.array([word_count, avg_word_length])])

    # Scale the features using the loaded StandardScaler
    X_scaled = scaler.transform(X_combined)

    # Make the prediction
    prediction = model_svc.predict(X_scaled)

    if int(prediction[0]) == 0:
        prediction = 'Human Generated'
    else:
        prediction = 'AI Generated'

    # Return the prediction in the response
    return render_template('index.html',result=prediction)

if __name__ == '__main__':
    app.run(port=5100, debug=True)