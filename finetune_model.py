# finetune_model.py

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch

class IMDbDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(train_file='train.csv', test_file='test.csv'):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = IMDbDataset(
        reviews=train_df.review.to_numpy(),
        labels=train_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=128
    )
    test_dataset = IMDbDataset(
        reviews=test_df.review.to_numpy(),
        labels=test_df.label.to_numpy(),
        tokenizer=tokenizer,
        max_length=128
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    model.save_pretrained('sentiment_model')
    tokenizer.save_pretrained('sentiment_model')

if __name__ == '__main__':
    train_model()
