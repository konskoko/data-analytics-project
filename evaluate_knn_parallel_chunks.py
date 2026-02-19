import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from time import perf_counter
from joblib import Parallel, delayed
from config import MAX_FEATURES, RANDOM_STATE
from data_utils import get_data
from scipy.stats import mode

CHUNK_SIZE = 2000

def _fill_jaccard_chunk(start, end, X, XT, cardinality, out_matrix):
    X_slice = X[start:end]
    card_slice = cardinality[start:end]

    intersection = X_slice.dot(XT)
    intersection = intersection.toarray().astype(np.float32)

    union = card_slice[:, None] + cardinality[None, :] - intersection

    intersection = np.where(union == 0, 1.0, intersection / union)
    intersection *= -1
    intersection += 1.0

    out_matrix[start:end] = intersection.astype(np.float16)

def compute_jaccard_matrix(X, n_jobs=-1):

    cardinality = np.array(X.multiply(X).sum(axis=1)).flatten().astype(np.float32)

    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float16)

    XT = X.T
    slices = [(i, min(i + CHUNK_SIZE, n_samples)) for i in range(0, n_samples, CHUNK_SIZE)]

    Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fill_jaccard_chunk)(
            start, end, X, XT, cardinality, dist_matrix
        )
        for start, end in slices
    )

    return dist_matrix

def main():
    print("Loading data...")
    t0 = perf_counter()
    X, y = get_data(preprocess=False)
    t1 = perf_counter()
    print(f"Data loaded in {t1 - t0:.4f} seconds")
    print(f"Dataset shape: {len(X)}")

    print("Extracting features...")
    t0 = perf_counter()
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, binary=True, dtype=np.uint8)
    X_bow = vectorizer.fit_transform(X)
    t1 = perf_counter()
    print(f"Feature extraction finished in {t1 - t0:.4f} seconds")

    print("Precomputing Jaccard distance matrix...")
    t0 = perf_counter()
    dist_matrix = compute_jaccard_matrix(X_bow)
    t1 = perf_counter()
    print(f"Distance matrix computed in {t1 - t0:.4f} seconds")

    print("Running KNN CV with precomputed distances...")
    t0 = perf_counter()
    stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    knn_scores = []

    # Encode labels to integers
    y = y.reset_index(drop=True)
    y_values = y.values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_values)

    n_neighbors = 5

    for fold_idx, (train_idx, test_idx) in enumerate(stratified_cv.split(dist_matrix, y)):
        fold_t0 = perf_counter()

        # Isolate distances of test set
        dists_test_all = dist_matrix[test_idx, :]
        dists_test_train = dists_test_all[:, train_idx]

        # Find k nearest neighbors
        knn_indices = np.argpartition(dists_test_train, n_neighbors, axis=1)[:, :n_neighbors]

        # Get labels of neighbors
        y_train_subset = y_encoded[train_idx]
        nearest_labels = y_train_subset[knn_indices]

        # Majority Vote
        predictions_encoded, _ = mode(nearest_labels, axis=1)

        # Accuracy
        y_test_subset = y_encoded[test_idx]
        score = np.mean(predictions_encoded == y_test_subset)

        knn_scores.append(score)
        print(f"Fold {fold_idx+1} accuracy: {score:.4f} (Time: {perf_counter() - fold_t0:.2f}s)")

    knn_scores = np.array(knn_scores)
    t1 = perf_counter()

    print(f"\nTotal CV time: {t1 - t0:.4f} seconds")
    print(f"KNN Accuracy: {knn_scores.mean():.4f} (+/- {knn_scores.std():.4f})")

if __name__ == "__main__":
    main()
