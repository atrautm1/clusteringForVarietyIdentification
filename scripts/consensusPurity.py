from scipy.spatial.distance import hamming
from hdbscan import HDBSCAN
from scipy.stats import entropy
import pandas as pd
import logging
import numpy as np
from scipy.spatial.distance import cdist


def calculate_purity_matrix(raw, consensus):

    # cdist with 'hamming' returns the proportion of mismatching coordinates (0.0 to 1.0)
    hamming_distances = cdist(raw, consensus, metric="hamming")

    # Purity percentage is (1 - hamming_distance) * 100
    purity_matrix = (1.0 - hamming_distances) * 100

    return purity_matrix


def calculate_purity_by_name(line_series, consensus_series):
    """
    Compares a line's genotypes against a consensus cluster and
    returns the purity percentage.
    """
    # Filter out missing data/NaNs so we only compare overlapping valid markers
    valid_mask = line_series.notna() & consensus_series.notna()
    line_valid = line_series[valid_mask]
    consensus_valid = consensus_series[valid_mask]

    if len(line_valid) == 0:
        return 0.0  # No overlapping markers

    # 1.0 - hamming distance = proportion of identical markers
    purity_proportion = 1.0 - hamming(line_valid, consensus_valid)
    return purity_proportion * 100


def optimize_hdbscan(embedding):
    best_score = -1
    best_size = 5

    # Test min_cluster_sizes from 5 to 50
    for size in range(5, 51, 5):
        clusterer = HDBSCAN(min_cluster_size=size).fit(embedding)

        # HDBSCAN provides a built-in validity metric for density
        score = clusterer.relative_validity_

        if score > best_score:
            best_score = score
            best_size = size

    print(f"Optimal min_cluster_size found: {best_size} (Score: {best_score:.3f})")
    return best_size


def get_cluster_metrics(data, total_class_counts):
    def calculate_cluster_metrics(cluster_df):
        cluster_size = len(cluster_df)

        # 1. Replace whitespace/empty strings with NaN, then drop all NaNs
        valid_varieties = (
            cluster_df["variety"].replace(r"^\s*$", np.nan, regex=True).dropna()
        )

        ignored_classes = ["Group 0", 0, "0", "Unknown", "Unlabeled"]
        valid_varieties = valid_varieties[~valid_varieties.isin(ignored_classes)]

        class_counts = valid_varieties.value_counts()

        # 3. Handle clusters with no valid non-ignored varieties
        if class_counts.empty:
            majority_class = "Unlabeled"
            majority_count = 0
            purity = 0.0
            ent = 0.0
            recall = 0.0
        else:
            majority_class = class_counts.idxmax()
            majority_count = class_counts.max()

            purity = (majority_count / cluster_size) * 100
            ent = entropy(class_counts, base=2)

            total_majority_in_dataset = total_class_counts.get(majority_class, 0)
            recall = (
                (majority_count / total_majority_in_dataset) * 100
                if total_majority_in_dataset > 0
                else 0.0
            )

        return pd.Series(
            {
                "Size": cluster_size,
                "Majority_Class": majority_class,
                "Purity_%": purity,
                "Entropy": ent,
                "Recall_%": recall,
            }
        )

    # 4. Group by the Cluster Label and apply the function
    cluster_summary = (
        data.groupby("cluster").apply(calculate_cluster_metrics).reset_index()
    )

    # Round the numbers for clean reading
    cluster_summary = cluster_summary.round(
        {"Purity_%": 2, "Entropy": 3, "Recall_%": 2}
    )

    logging.info(cluster_summary.to_string(index=False))
    return cluster_summary
