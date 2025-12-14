import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
from config import DATA_PATH, N_SAMPLES_DEV


def preprocess_text(text: str):
    punct_remover = str.maketrans('', '',string.punctuation)
    text_no_punct = text.strip().lower().translate(punct_remover)
    stop_words = set(stopwords.words('english'))

    words = [word for word in text_no_punct.split() if word not in stop_words]
    processed_text =' '.join(words)
    return processed_text

def get_data():
    # Read data
    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    if N_SAMPLES_DEV is not None:
        train_df = train_df.loc[:N_SAMPLES_DEV, :]

    train_df.loc[:, ['Title', 'Content']] = train_df[['Title', 'Content']].map(preprocess_text)
    X = train_df['Title'] + train_df['Content']
    y = train_df['Label']

    return X, y