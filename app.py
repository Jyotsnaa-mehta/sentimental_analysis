import streamlit as st
import joblib
import sqlite3
import pandas as pd
from datetime import datetime

# Load trained model
pipe = joblib.load("sentiment_model.joblib")

# App configuration
st.set_page_config(page_title="AI Sentiment Analyzer", page_icon="💬", layout="centered")

# Title and description
st.title("💬 AI Sentiment Analyzer")
st.write("Type any review or feedback below to see if it's **positive** or **negative**.")

# Input text
text_input = st.text_area("Enter text here:")

# Database setup
conn = sqlite3.connect("sentiments.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    sentiment TEXT,
    timestamp TEXT
)
""")
conn.commit()

# Analyze button
if st.button("Analyze Sentiment"):
    if text_input.strip() != "":
        # Predict sentiment
        prediction = pipe.predict([text_input])[0]
        label = "Positive 😄" if prediction == 1 else "Negative 😞"

        # Save to database
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO feedback (text, sentiment, timestamp) VALUES (?, ?, ?)",
                       (text_input, label, timestamp))
        conn.commit()

        # Show prediction
        st.markdown(f"### Prediction: {label}")
    else:
        st.warning("Please enter some text before analyzing!")

# Show saved predictions
st.write("---")
st.subheader("📊 Saved Predictions")
df = pd.read_sql_query("SELECT * FROM feedback ORDER BY id DESC", conn)
st.dataframe(df)

# Clear button
st.write("---")
if st.button("🗑️ Clear All Saved Predictions"):
    cursor.execute("DELETE FROM feedback")
    conn.commit()
    st.success("All records deleted successfully!")

conn.close()

# Footer
st.write("---")
st.caption("✨ Developed with Streamlit + Scikit-learn | Your AI Sentiment Assistant")
