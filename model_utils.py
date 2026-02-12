from time import perf_counter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import RANDOM_STATE, N_JOBS

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
