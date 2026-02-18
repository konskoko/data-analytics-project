from time import perf_counter
import sklearn.svm
import sklearn.feature_extraction
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from config import RANDOM_STATE, MAX_FEATURES
from data_utils import get_data
from model_utils import evaluate_pipeline


def main():
    t0 = perf_counter()
    X, y = get_data()

    t1 = perf_counter()
    print(f"Data loaded in {t1 - t0:.2f} seconds")
    print(f"{len(X)} training samples")
    print("-" * 30)

    svm_pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES),
        MaxAbsScaler(),
        sklearn.svm.LinearSVC(loss='hinge', max_iter=20000, random_state=RANDOM_STATE)
    )

    evaluate_pipeline(svm_pipe, X, y, "SVM")


if __name__ == "__main__":
    main()
