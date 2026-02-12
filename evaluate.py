from time import perf_counter
import sklearn.svm
import sklearn.ensemble
import sklearn.feature_extraction
import sklearn.neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from config import RANDOM_STATE, MAX_FEATURES, N_JOBS
from data_utils import get_data


def jaccard_distance(x, y):
    """
    Jaccard distance for sparse matrices.
    """
    intersection = x.dot(y.T)[0, 0]
    union = x.getnnz() + y.getnnz() - intersection
    return 1.0 - intersection / union if union > 0 else 0.0


def evaluate_pipeline(pipeline, X, y, model_name, k=5):
    """
    Evaluate a model pipeline using Stratified K-Fold Cross-Validation.
    """
    print(f"Starting evaluation for {model_name}...")
    t0 = perf_counter()

    stratified_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(pipeline, X, y, cv=stratified_cv, n_jobs=N_JOBS, verbose=1)

    t1 = perf_counter()
    duration = t1 - t0

    print(f"{model_name} accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"{model_name} evaluation finished in {duration:.2f} seconds\n")
    return scores


def main():
    t0 = perf_counter()
    X, y = get_data()


    t1 = perf_counter()
    print(f"Data loaded in {t1 - t0:.2f} seconds")
    print(f"{len(X)} training samples")
    print("-" * 30)

    pipelines = []

    svm_pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES),
        MaxAbsScaler(),
        sklearn.svm.LinearSVC(loss='hinge', max_iter=10000, random_state=RANDOM_STATE)
    )
    pipelines.append(("SVM", svm_pipe))

    knn_pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES, binary=True),
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric=jaccard_distance, n_jobs=1)
    )
    pipelines.append(("KNN", knn_pipe))

    for name, pipe in pipelines:
        evaluate_pipeline(pipe, X, y, name)


if __name__ == "__main__":
    main()
