import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Function to analyze sentiment
def analyze_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).item()
    
    # Convert predictions to human-readable labels
    sentiment = 'Positive' if predictions == 1 else 'Negative'
    return sentiment

# Streamlit app
st.title("Sentiment Analysis")

user_input = st.text_area("Enter text here...")

if st.button("Analyze"):
    if user_input:
        sentiment = analyze_sentiment(user_input)
        st.write(f"Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
