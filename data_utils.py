import string
from time import perf_counter
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from wordcloud import STOPWORDS
from config import DATA_PATH, DEV_DATA_FRACTION, RANDOM_STATE


_STOP_WORDS = set(stopwords.words('english'))
_STOP_WORDS.update(STOPWORDS)
_STOP_WORDS.update(["said", "say", "may", "one"])
_LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str, tokenizer='simple_rem_punct'):
    if tokenizer == 'simple':
        tokens = text.strip().lower().split()
    elif tokenizer == 'simple_rem_punct':
        tokens = text.strip().lower().translate(
            str.maketrans('', '', string.punctuation)
        ).split()
    elif tokenizer == 'nltk':
        tokens = nltk.word_tokenize(text.strip().lower())
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    processed_tokens = [
        _LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in _STOP_WORDS
    ]

    return ' '.join(processed_tokens)

def get_data(sample=True, preprocess=False):
    t0 = perf_counter()

    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    if sample and DEV_DATA_FRACTION is not None:
        train_df = train_df.sample(frac=DEV_DATA_FRACTION, random_state=RANDOM_STATE)

    train_df['text'] = train_df['Title'] + " " + train_df['Content']
    if preprocess:
        train_df['text'] = train_df['text'].map(preprocess_text)
    t1 = perf_counter()
    print(f"Loaded and preprocessed {len(train_df)} samples in {t1 - t0:.2f} seconds")
    X = train_df['text']
    y = train_df['Label']

    return X, y

def get_test_data():
    test_df = pd.read_csv(DATA_PATH / 'test_without_labels.csv')
    test_df['text'] = test_df['Title'] + " " + test_df['Content']
    test_df['text'] = test_df['text'].map(preprocess_text)
    X_test = test_df['text']
    return test_df, X_test