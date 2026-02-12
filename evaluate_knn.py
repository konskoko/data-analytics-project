from time import perf_counter
import sklearn.neighbors
import sklearn.feature_extraction
from sklearn.pipeline import make_pipeline
from config import MAX_FEATURES
from data_utils import get_data
from model_utils import evaluate_pipeline, jaccard_distance


def main():
    t0 = perf_counter()
    X, y = get_data()

    t1 = perf_counter()
    print(f"Data loaded in {t1 - t0:.2f} seconds")
    print(f"{len(X)} training samples")
    print("-" * 30)

    knn_pipe = make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES, binary=True),
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric=jaccard_distance, n_jobs=1)
    )

    evaluate_pipeline(knn_pipe, X, y, "KNN")


if __name__ == "__main__":
    main()
