import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from config import DATA_PATH, DEV_DATA_FRACTION, RANDOM_STATE


def preprocess_text(text: str):
    punct_remover = str.maketrans('', '',string.punctuation)
    text_no_punct = text.strip().lower().translate(punct_remover)
    stop_words = set(stopwords.words('english'))

    words = [word for word in text_no_punct.split() if word not in stop_words]
    processed_text = ' '.join(words)
    return processed_text

def get_data(sample=True):
    # Read data
    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    if sample and DEV_DATA_FRACTION is not None:
        train_df = train_df.sample(frac=DEV_DATA_FRACTION, random_state=RANDOM_STATE)

    train_df.loc[:, ['Title', 'Content']] = train_df[['Title', 'Content']].map(preprocess_text)
    X = train_df['Title'] + " " + train_df['Content']
    y = train_df['Label']

    return X, y

def get_test_data():
    test_df = pd.read_csv(DATA_PATH / 'test_without_labels.csv')
    test_df.loc[:, ['Title', 'Content']] = test_df[['Title', 'Content']].map(preprocess_text)
    X_test = test_df['Title'] + " " + test_df['Content']
    return test_df, X_test