from time import perf_counter
import sklearn.svm
import sklearn.feature_extraction
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from config import RANDOM_STATE, MAX_FEATURES
from data_utils import get_data, get_test_data
from model_utils import output_preds

def get_svm_pipeline():
    return make_pipeline(
        sklearn.feature_extraction.text.CountVectorizer(max_features=MAX_FEATURES),
        MaxAbsScaler(),
        sklearn.svm.LinearSVC(loss='hinge', max_iter=10000, random_state=RANDOM_STATE)
    )

def main():
    print("Loading training data...")
    t0 = perf_counter()
    X_train, y_train = get_data(sample=False)
    t1 = perf_counter()
    print(f"Training data loaded in {t1 - t0:.2f} seconds")

    print("Loading test data...")
    test_df, X_test = get_test_data()

    print("Training SVM model...")
    t0 = perf_counter()
    pipeline = get_svm_pipeline()
    pipeline.fit(X_train, y_train)
    t1 = perf_counter()
    print(f"Model trained in {t1 - t0:.2f} seconds")

    print("Generating predictions...")
    t0 = perf_counter()
    y_pred = pipeline.predict(X_test)
    t1 = perf_counter()
    print(f"Predictions generated in {t1 - t0:.2f} seconds")

    output_preds(test_df, y_pred, "SVM")

if __name__ == "__main__":
    main()
