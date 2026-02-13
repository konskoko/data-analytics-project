from time import perf_counter
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from config import RANDOM_STATE, N_JOBS

def evaluate_pipeline(pipeline, X, y, model_name, k=5):
    """
    Evaluate a model pipeline using Stratified K-Fold Cross-Validation.
    """
    print(f"Starting evaluation for {model_name}...")
    t0 = perf_counter()

    stratified_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(pipeline, X, y, cv=stratified_cv, n_jobs=N_JOBS, verbose=3)

    t1 = perf_counter()
    duration = t1 - t0

    print(f"{model_name} accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"{model_name} evaluation finished in {duration:.2f} seconds\n")
    return scores

def output_preds(test_df: pd.DataFrame, y_pred: pd.DataFrame, model_name: str):
    output_dir = Path('./test_predictions')
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file_name = f'test_preds_{model_name}_{timestamp}.csv'
    output_path = output_dir / output_file_name

    output_test_df = test_df.copy()
    output_test_df['Label'] = y_pred
    output_test_df[['Id', 'Label']].to_csv(
        output_path,
        index=False
    )
    print(f"Predictions saved to {output_path}")
