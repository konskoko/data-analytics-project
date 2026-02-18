from wordcloud import WordCloud
from pathlib import Path
from data_utils import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def generate_wordclouds():
    print("Loading data...")
    X, y = get_data(sample=False, preprocess=True)

    output_dir = Path('./wordclouds')
    output_dir.mkdir(exist_ok=True)

    categories = y.unique()

    print("Vectorizing data...")
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()

    print(f"Found {len(categories)} categories: {categories}")

    for category in categories:
        print(f"Generating wordcloud for category: {category}")

        mask = (y == category).values
        X_cat = X_tfidf[mask]
        word_scores = np.asarray(X_cat.sum(axis=0)).flatten()
        frequencies = {word: score for word, score in zip(feature_names, word_scores) if score > 0 and len(word) >= 3}

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequencies)

        filename = output_dir / f"wordcloud_{category}.png"
        wordcloud.to_file(filename)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    generate_wordclouds()
