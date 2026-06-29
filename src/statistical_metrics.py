import pandas as pd

from pymdma.tabular.measures.synthesis_val import (
    StatisticalSimScore,
    StatisticalDivergenceScore,
    Density,
    Coverage,
    CoherenceScore,
    Authenticity
)

def compare_numerical_stats(real_df, synthetic_df, numerical_cols):
    results = []

    for column in numerical_cols:
        mean_real = real_df[column].mean()
        mean_synthetic = synthetic_df[column].mean()
        abs_diff = abs(mean_real - mean_synthetic)

        results.append({
            "column": column,
            "mean_real": mean_real,
            "mean_synthetic": mean_synthetic,
            "absolute_difference": abs_diff
        })

    return pd.DataFrame(results)

def compare_categorical_stats(real_df, synthetic_df, categorical_cols):
    results = []

    for column in categorical_cols:
        proportions_real = real_df[column].value_counts(normalize=True)
        proportions_synthetic = synthetic_df[column].value_counts(normalize=True)

        all_categories = proportions_real.index.union(proportions_synthetic.index)

        proportions_real = proportions_real.reindex(all_categories, fill_value=0)
        proportions_synthetic = proportions_synthetic.reindex(all_categories, fill_value=0)

        abs_diff = abs(proportions_real - proportions_synthetic).mean()

        results.append({
            "column": column,
            "mean_absolute_proportions_difference": abs_diff
        })

    return pd.DataFrame(results)

def build_col_map(numerical_cols, categorical_cols):
    col_map = {}

    for column in numerical_cols:
        col_map[column] = {"type": {"tag": "continuous"}}

    for column in categorical_cols:
        col_map[column] = {"type": {"tag": "discrete"}}

    return col_map

def statistical_similarity(real_df, synthetic_df, col_map):
    metric = StatisticalSimScore(col_map=col_map)
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result

def statistical_divergence(real_df, synthetic_df, col_map, score="js"):
    metric = StatisticalDivergenceScore(col_map=col_map, score=score)
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result

def density(real_df, synthetic_df):
    metric = Density()
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result

def coverage(real_df, synthetic_df):
    metric = Coverage()
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result

def coherence(real_df, synthetic_df):
    metric = CoherenceScore()
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result

def authenticity(real_df, synthetic_df):
    metric = Authenticity()
    result = metric.compute(real_df.to_numpy(), synthetic_df.to_numpy())
    return result