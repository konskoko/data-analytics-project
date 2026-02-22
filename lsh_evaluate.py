import numpy as np
import pandas as pd
from time import perf_counter
from sklearn.feature_extraction.text import CountVectorizer
from datasketch import MinHash, MinHashLSH
from data_utils import get_data, get_test_data
from config import MAX_FEATURES

K_NEIGHBORS = 7
THRESHOLDS = [0.4, 0.7, 0.9]
PERMUTATIONS = [16, 32, 64, 128, 256]
CHUNK_SIZE = 2000

def compute_exact_jaccard_topk(X_train_bow, X_test_bow):
    n_test = X_test_bow.shape[0]

    topk_indices = np.zeros((n_test, K_NEIGHBORS), dtype=np.int32)

    card_train = np.array(X_train_bow.sum(axis=1)).flatten().astype(np.float32)
    X_train_T = X_train_bow.T

    for start in range(0, n_test, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_test)
        X_test_chunk = X_test_bow[start:end]
        card_test_chunk = np.array(X_test_chunk.sum(axis=1)).flatten().astype(np.float32)

        intersection = X_test_chunk.dot(X_train_T).toarray().astype(np.float32)
        union = card_test_chunk[:, None] + card_train[None, :] - intersection

        jaccard_dist = np.where(union == 0, 0.0, 1.0 - intersection / union)

        chunk_topk = np.argpartition(jaccard_dist, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
        topk_indices[start:end] = chunk_topk

    return topk_indices

def create_minhashes_from_sparse(X_bow, feature_names, num_perm):
    minhashes = []
    for i in range(X_bow.shape[0]):
        m = MinHash(num_perm=num_perm)
        non_zero_cols = X_bow[i].nonzero()[1]
        for col in non_zero_cols:
            m.update(feature_names[col].encode('utf8'))
        minhashes.append(m)
    return minhashes

def compute_jaccard_distance_sparse(row_a, row_b):
    intersection = row_a.multiply(row_b).nnz
    union = row_a.nnz + row_b.nnz - intersection
    if union == 0:
        return 0.0
    return 1.0 - intersection / union

def run_evaluation():
    print("Loading data...")
    X_train, y_train = get_data(sample=True, preprocess=True)
    test_meta, X_test = get_test_data()

    print("\n--- Running Brute-Force K-NN ---")
    vectorizer = CountVectorizer(max_features=MAX_FEATURES, binary=True, dtype=np.uint8)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    feature_names = vectorizer.get_feature_names_out()

    bf_start = perf_counter()
    bf_topk_indices = compute_exact_jaccard_topk(X_train_bow, X_test_bow)
    bf_query_time = perf_counter() - bf_start
    print(f"Brute-Force Query Time: {bf_query_time:.2f}s")

    results = [{
        "Type": "Brute-Force-Jaccard",
        "BuildTime": 0.0,
        "QueryTime": round(bf_query_time, 2),
        "Total Time": round(bf_query_time, 2),
        "Fraction of true K nearest": "100%",
        "Parameters": f"K={K_NEIGHBORS}"
    }]

    for num_perm in PERMUTATIONS:
        print(f"\n--- Building MinHashes with Permutations={num_perm} ---")

        train_mh_start = perf_counter()
        train_minhashes = create_minhashes_from_sparse(X_train_bow, feature_names, num_perm)
        train_mh_time = perf_counter() - train_mh_start

        test_mh_start = perf_counter()
        test_minhashes = create_minhashes_from_sparse(X_test_bow, feature_names, num_perm)
        test_mh_time = perf_counter() - test_mh_start

        print(f"Train MinHash time: {train_mh_time:.2f}s, Test MinHash time: {test_mh_time:.2f}s")

        for threshold in THRESHOLDS:
            print(f"\n--- LSH: Permutations={num_perm}, Threshold={threshold} ---")

            # Build LSH Index
            build_start = perf_counter()
            try:
                lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            except ValueError:
                print(f"Invalid combination: threshold={threshold}, num_perm={num_perm}, skipping...")
                continue
            for i, m in enumerate(train_minhashes):
                lsh.insert(i, m)
            build_time = perf_counter() - build_start + train_mh_time
            print(f"LSH Build Time: {build_time:.2f}s")

            # Query Index
            query_start = perf_counter()
            lsh_topk_indices = []
            for i, test_m in enumerate(test_minhashes):
                candidates = lsh.query(test_m)

                if len(candidates) <= K_NEIGHBORS:
                    lsh_topk_indices.append(candidates)
                    continue

                # If more candidates than K, keep the K with smallest Jaccard distance
                dists = np.array([compute_jaccard_distance_sparse(X_test_bow[i], X_train_bow[idx]) for idx in candidates])
                top_k_pos = np.argpartition(dists, K_NEIGHBORS)[:K_NEIGHBORS]
                lsh_topk_indices.append([candidates[p] for p in top_k_pos])

            query_time = perf_counter() - query_start + test_mh_time
            print(f"LSH Query Time: {query_time:.2f}s")

            # Fraction of true K most similar documents found by LSH
            matches = 0
            total_possible = X_test_bow.shape[0] * K_NEIGHBORS
            for bf_indices, lsh_indices in zip(bf_topk_indices, lsh_topk_indices):
                matches += len(set(bf_indices).intersection(set(lsh_indices)))

            fraction_k = matches / total_possible if total_possible > 0 else 0
            print(f"Fraction of true K nearest: {fraction_k:.2%}")

            results.append({
                "Type": "LSH-Jaccard",
                "BuildTime": round(build_time, 2),
                "QueryTime": round(query_time, 2),
                "Total Time": round(build_time + query_time, 2),
                "Fraction of true K nearest": f"{fraction_k:.2%}",
                "Parameters": f"Perm={num_perm}, K={K_NEIGHBORS}, T={threshold}"
            })

    results_df = pd.DataFrame(results)
    print("\n--- Evaluation Results ---")
    print(results_df.to_string(index=False))

    out_path = 'lsh_evaluate_results.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()