import pandas as pd
import joblib
import os
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from ocr_extraction import extract_text
from tqdm import tqdm  

dataset_path = "docs-sm/"
categories = ["invoice", "budget", "email"]  
data = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    for filename in tqdm(os.listdir(category_path), desc=f"Processing {category}", unit="file"):
        file_path = os.path.join(category_path, filename)
        text = extract_text(file_path) 
        data.append((text, category))

df = pd.DataFrame(data, columns=["text", "category"])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    return text.strip()

df["cleaned_text"] = df["text"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["category"], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)
joblib.dump(model, "models/classifier.pkl")

y_pred = model.predict(X_test)
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
