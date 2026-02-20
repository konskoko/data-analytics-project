import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    cost_grid = (s1[:, None] - s2[None, :]) ** 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_grid[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])

    return np.sqrt(dtw_matrix[n, m])


def main():
    start_time = time.perf_counter()

    df = pd.read_csv('data/dtw_test.csv')

    series_a = [np.array(json.loads(s), dtype=np.float64) for s in df['series_a']]
    series_b = [np.array(json.loads(s), dtype=np.float64) for s in df['series_b']]

    distances = []
    for s1, s2 in tqdm(zip(series_a, series_b), total=len(df)):
        distances.append(dtw_distance(s1, s2))

    execution_time = time.perf_counter() - start_time
    print(f"Total time: {execution_time:.2f}s")

    df['dtw_distance'] = distances
    df[['id', 'dtw_distance']].to_csv('dtw_results.csv', index=False)
    print("Results saved to dtw_results.csv")


if __name__ == "__main__":
    main()
