# inference.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentModel:
    def __init__(self, model_path='sentiment_model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return 'positive' if predicted_class == 1 else 'negative'

if __name__ == '__main__':
    model = SentimentModel()
    text = "The movie was fantastic!"
    print(model.predict(text))
