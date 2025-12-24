
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import sklearn
import sklearn.feature_extraction
from time import perf_counter
from config import RANDOM_STATE
from data_utils import get_data

def k_fold_cv(model=None, X=None, y=None, k=5):
    stratified_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features = 10000),
        MaxAbsScaler(),
        model
    )
    score = cross_val_score(pipe, X, y, cv=stratified_cv, n_jobs=-1, verbose=3)

    return score

def main():
    t0 = perf_counter()
    X, y = get_data()
    
    t1 = perf_counter()
    print(f"Data loaded in {t1 - t0} seconds")
    print(f"{X.shape[0]} training samples")

    # SVM on BOW
    t0 = perf_counter()
    # svm_clf = sklearn.svm.SVC(kernel='linear')
    # Use hinge loss for standard SVM
    # Increase max_iter to avoid convergence issues
    svm_clf = sklearn.svm.LinearSVC(loss='hinge', max_iter=10000, random_state=RANDOM_STATE)
    svm_score = k_fold_cv(svm_clf, X, y)
    print(f"SVM accuracy: {svm_score.mean()} with std {svm_score.std()}")
    t1 = perf_counter()
    print(f"SVM on BOW finished in {t1 - t0} seconds")

    # KNN on BOW
    # TODO: metric should be Jaccard
    t0 = perf_counter()
    knn_clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
    knn_score = k_fold_cv(knn_clf, X, y)
    print(f"KNN accuracy: {knn_score.mean()} with std {knn_score.std()}")
    t1 = perf_counter()
    print(f"KNN on BOW finished in {t1 - t0} seconds")


if __name__ == "__main__":
    main()