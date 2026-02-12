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


def evaluate_model(model, X, y, model_name, k=5):
    """
    Evaluate a model using Stratified K-Fold Cross-Validation.
    """
    print(f"Starting evaluation for {model_name}...")
    t0 = perf_counter()

    stratified_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES),
        MaxAbsScaler(),
        model
    )

    scores = cross_val_score(pipe, X, y, cv=stratified_cv, n_jobs=N_JOBS, verbose=1)

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

    models = [
        # Use hinge loss for standard SVM
        # Increase max_iter to avoid convergence issues
        # svm_clf = sklearn.svm.SVC(kernel='linear') ??
        ("SVM", sklearn.svm.LinearSVC(loss='hinge', max_iter=10000, random_state=RANDOM_STATE)),
        # TODO: metric should be Jaccard
        ("KNN", sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric='minkowski'))
    ]

    for name, model in models:
        evaluate_model(model, X, y, name)


if __name__ == "__main__":
    main()
