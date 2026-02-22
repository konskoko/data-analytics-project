from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import sklearn
from config import RANDOM_STATE
from data_utils import get_data
from model_utils import evaluate_pipeline


def main():
    X, y = get_data(preprocess=True)

    svm_tfidf_pipe = make_pipeline(
        TfidfVectorizer(max_features=200000, ngram_range=(1, 2)),
        sklearn.svm.LinearSVC(loss='hinge', max_iter=10000, random_state=RANDOM_STATE)
    )

    evaluate_pipeline(svm_tfidf_pipe, X, y, "SVM_TF-IDF")


if __name__ == "__main__":
    main()
