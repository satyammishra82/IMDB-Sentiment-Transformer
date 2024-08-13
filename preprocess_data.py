# preprocess_data.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_dir='aclImdb'):
    train_dir = os.path.join(data_dir, 'aclImdb', 'train')
    test_dir = os.path.join(data_dir, 'aclImdb', 'test')

    def read_data(data_type):
        pos_dir = os.path.join(data_type, 'pos')
        neg_dir = os.path.join(data_type, 'neg')

        data = []
        for label, dir_name in enumerate([pos_dir, neg_dir]):
            for file_name in os.listdir(dir_name):
                if file_name.endswith('.txt'):
                    with open(os.path.join(dir_name, file_name), 'r', encoding='utf-8') as file:
                        data.append((file.read(), label))
        return data

    train_data = read_data(train_dir)
    test_data = read_data(test_dir)

    train_df = pd.DataFrame(train_data, columns=['review', 'label'])
    test_df = pd.DataFrame(test_data, columns=['review', 'label'])

    return train_df, test_df

def preprocess_data(train_df, test_df, max_samples=10000):
    train_df = train_df.sample(max_samples // 2)
    test_df = test_df.sample(max_samples // 2)
    
    return train_df, test_df

if __name__ == '__main__':
    train_df, test_df = load_data()
    train_df, test_df = preprocess_data(train_df, test_df)
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
