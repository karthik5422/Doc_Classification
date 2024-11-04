from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import sqlite3
import os
import random
import PyPDF2
from datetime import datetime

from extract_text import extract_text_from_pdf

app = Flask(__name__)

MODEL_PATH = "models/text_classification_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_model(MODEL_PATH, VECTORIZER_PATH)

DATABASE = "meta_data.db"

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class TEXT,
            metadata TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def extract_warranty_data(text):
    return {"warranty_info": "Warranty extracted details"}

def extract_transaction_data(text):
    return {"transaction_info": "Transaction extracted details"}

def extract_troubleshooting_data(text):
    return {"troubleshooting_info": "Troubleshooting extracted details"}

class_functions = {
    "Warranty": extract_warranty_data,
    "Transactions": extract_transaction_data,
    "Troubleshooting": extract_troubleshooting_data
}

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    pdf_path = data.get("text")
    pdf_text = extract_text_from_pdf(pdf_path)
    text = pdf_text
    model_path = data.get("model_path", MODEL_PATH)
    vectorizer_path = data.get("vectorizer_path", VECTORIZER_PATH)

    model, vectorizer = load_model(model_path, vectorizer_path)

    text_vectorized = vectorizer.transform([text])
    predicted_class = model.predict(text_vectorized)[0]

    extraction_function = class_functions.get(predicted_class)
    if extraction_function:
        metadata = extraction_function(text)
    else:
        return jsonify({"error": "No extraction function for this class"}), 400

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    random_id = random.randint(1, 1000000)  
    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("INSERT INTO metadata (id, class, metadata, timestamp) VALUES (?, ?, ?, ?)", 
                   (random_id, predicted_class, str(metadata), current_timestamp))
    
    conn.commit()
    conn.close()

    return jsonify({
        "predicted_class": predicted_class,
        "metadata": metadata
    })

if __name__ == '__main__':
    app.run(debug=True)
