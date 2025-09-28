import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

POS_KEYWORDS = {"good", "excellent", "great", "amazing"} 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
# Download NLTK stopwords if not already available
nltk.download("stopwords")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# Text preprocessing function
def clean_text(text):
    """Cleans text by lowercasing, removing stopwords, and applying stemming."""
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Remove non-alphabetic characters
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)





# Prediction function
def predict_sentiment(text):
    """Predicts sentiment for a given text."""
    
    # 1️⃣ Check short positive keywords first
    if text.lower().strip() in POS_KEYWORDS:
        return "Positive", 1.0  # return high confidence for keyword

    # 2️⃣ Clean text
    text = clean_text(text)

    # 3️⃣ Convert to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # 4️⃣ Handle unknown / empty sequence
    if len(sequence[0]) == 0:
        # Word not in tokenizer vocab
        return "Negative", 0.5  # or you can return "Negative", 0.5

    # 5️⃣ Pad sequence
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # 6️⃣ Predict
    prediction = model.predict(padded_sequence, verbose=0)[0][0]

    # 7️⃣ Return sentiment
    sentiment = "Positive" if prediction > 0.4 else "Negative"
    return sentiment, float(prediction)


# Home and prediction route
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")
        if text:
            result, confidence = predict_sentiment(text)

    return render_template(
        "index.html",
        result=result,
        confidence=round(confidence, 2) if confidence else None
    )


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
