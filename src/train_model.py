import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_PATH = "labeled_data.csv"   
MODEL_PATH = "models/text_classification_model.pkl"   
VECTORIZER_PATH = "models/vectorizer.pkl"   

def load_data(csv_path=DATA_PATH):
    """Load labeled text data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}")
    data = pd.read_csv(csv_path)
    data['text'] = data['text'].fillna("").str.lower()   
    return data

def train_model():
    data = load_data()
    X = data['text']
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
 
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
 
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_tfidf, y_train)
 
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
 
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_model()
