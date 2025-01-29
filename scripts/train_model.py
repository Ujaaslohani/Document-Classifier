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
from tqdm import tqdm  # Import tqdm

# Load dataset from image folder
dataset_path = "docs-sm/"
categories = ["invoice", "budget", "email"]  # Modify as needed
data = []

# Wrap the category loop with tqdm to show progress
for category in categories:
    category_path = os.path.join(dataset_path, category)
    
    # Using tqdm to show progress for each file being processed in the category
    for filename in tqdm(os.listdir(category_path), desc=f"Processing {category}", unit="file"):
        file_path = os.path.join(category_path, filename)
        text = extract_text(file_path)  # Extract text from image
        data.append((text, category))

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "category"])

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text.strip()

df["cleaned_text"] = df["text"].apply(clean_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["category"], test_size=0.2, random_state=42)

# Model Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("classifier", LogisticRegression())
])

# Train & Save Model
model.fit(X_train, y_train)
joblib.dump(model, "models/classifier.pkl")

# Evaluate Model
y_pred = model.predict(X_test)
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))
