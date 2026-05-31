import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("sentiment_dataset.csv")

# Select required columns
df = df[["text", "target"]]

# Convert labels
df["label"] = df["target"].map({0: 0, 4: 1})

# Remove missing values
df = df.dropna(subset=["text"])

# Features and Labels
X = df["text"]
y = df["label"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Pipeline
model = Pipeline([
    ("vectorizer", CountVectorizer(stop_words="english")),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save Model
joblib.dump(model, "sentiment_model.joblib")

# Test Samples
samples = [
    "I love this product",
    "This service is terrible",
    "The experience was average"
]

predictions = model.predict(samples)

for text, label in zip(samples, predictions):
    sentiment = "Positive" if label == 1 else "Negative"
    print(f"{text} --> {sentiment}")

  
  
    
