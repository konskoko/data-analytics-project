# Data Analytics Project

Text classification and time series similarity project. Includes SVM and KNN categorical classifiers, Locality Sensitive Hashing (LSH) for nearest neighbors, and a custom Dynamic Time Warping (DTW) implementation.

## Setup

Requires Python 3.13+ and the [uv](https://docs.astral.sh/uv/) package manager.

```bash
uv sync
```

The training data (`train.csv`) and test data (`test_without_labels.csv`) should be placed inside a `data/` directory at the project root for text classification.
For the time series similarity part, `dtw_test.csv` is expected in the `data/` directory.

## Project Structure

This repository contains the code used to generate the results in the project report. Below are the mappings from the report sections to the corresponding runnable scripts, allowing for easy testing.

### Part 1: Text Classification

| File | Description |
|---|---|
| `generate_wordclouds.py` | Generates per-category wordcloud images (Question 1.1). Saves PNGs to `wordclouds/`. |
| `evaluate_svm.py` | Evaluates the baseline SVM pipeline (BoW via `CountVectorizer` + `LinearSVC`). |
| `evaluate_knn_naive.py` | The naive KNN implementation using a custom Jaccard distance function. Very slow. |
| `evaluate_knn_parallel_chunks.py` | The optimized KNN implementation using chunked matrix multiplication and joblib parallelization. |
| `evaluate_svm_tfidf.py` | The best performing SVM pipeline using TF-IDF, bigrams, and title weighting. |

### Part 2: Nearest Neighbor Search (LSH)

| File | Description |
|---|---|
| `lsh_evaluate.py` | Compares the exact brute-force Jaccard search with an approximate approach based on Locality Sensitive Hashing (LSH) using MinHash signatures. Tests various thresholds and permutation counts. |

### Part 3: Time Series Similarity (DTW)

| File | Description |
|---|---|
| `dtw_calculator.py` | Core exact Dynamic Time Warping (DTW) implementation from scratch, accelerated using Numba. Computes distances and saves to `dtw.csv`. |

### Utilities

| File | Description |
|---|---|
| `data_utils.py` | Data loading and text preprocessing (tokenization, stopword removal, lemmatization with title weighting). |
| `model_utils.py` | Shared model utilities (stratified K-Fold CV, prediction output). |
| `produce_predictions.py` | Trains the final SVM pipeline on the full training set and outputs test predictions. |
| `eda.ipynb` | Exploratory data analysis notebook for inspecting the dataset. |
| `config.py` | Central configuration variables for paths and global parameters. |

## Running the Code

All scripts can be executed via `uv run`, for example:

```bash
# Wordclouds
uv run generate_wordclouds.py

# Model evaluations
uv run evaluate_svm.py
uv run evaluate_knn_parallel_chunks.py
uv run evaluate_svm_tfidf.py

# LSH evaluation
uv run lsh_evaluate.py

# DTW calculations
uv run dtw_calculator.py
```