import pandas as pd
from statistical_metrics import (
    compare_numerical_stats,
    compare_categorical_stats,
    build_col_map,
    statistical_similarity,
    statistical_divergence,
    density,
    coverage,
    coherence,
    authenticity
)

noise_levels = [0, 10, 20, 30, 40, 50]
models = ["GCS", "CTGAN", "TVAE"]
results = []

numerical_cols = [
    "age", "wtkg", "karnof", "cd40", "cd420", "cd80", "cd820"
]

categorical_cols = [
    "trt", "hemo", "homo", "drugs", "oprior",
    "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "treat", "offtrt"
]

selected_cols = numerical_cols + categorical_cols

for model in models:
    for noise in noise_levels:
        if noise == 0:
            real_df = pd.read_csv("second_dataset_train.csv")
        else:
            real_df = pd.read_csv(f"second_dataset_train_noisy_{noise}.csv")

        if model == "CTGAN":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_CTGAN_{noise}.csv")
        elif model == "GCS":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_GCS_{noise}.csv")
        elif model == "TVAE":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_TVAE_{noise}.csv")

        real_subset = real_df[selected_cols]
        synthetic_subset = synthetic_df[selected_cols]

        real_num_subset = real_df[numerical_cols]
        synthetic_num_subset = synthetic_df[numerical_cols]

        col_map = build_col_map(numerical_cols, categorical_cols)
        numerical_col_map = build_col_map(numerical_cols, [])

        numerical_results = compare_numerical_stats(real_subset, synthetic_subset, numerical_cols)
        categorical_results = compare_categorical_stats(real_subset, synthetic_subset, categorical_cols)
        similarity_result = statistical_similarity(real_subset, synthetic_subset, col_map)
        divergence_result = statistical_divergence(real_num_subset, synthetic_num_subset, numerical_col_map, score="js")
        density_result = density(real_num_subset, synthetic_num_subset)
        coverage_result = coverage(real_num_subset, synthetic_num_subset)
        coherence_result = coherence(real_num_subset, synthetic_num_subset)
        authenticity_result = authenticity(real_num_subset, synthetic_num_subset)

        results.append({
            "model": model,
            "noise": noise,
            "similarity_mean": similarity_result.dataset_level.stats["mean"],
            "similarity_std": similarity_result.dataset_level.stats["std"],
            "divergence_js_mean": divergence_result.dataset_level.stats["js_mean"],
            "divergence_js_std": divergence_result.dataset_level.stats["js_std"],
            "numerical_mean_abs_diff": numerical_results["absolute_difference"].mean(),
            "categorical_mean_abs_prop_diff": categorical_results["mean_absolute_proportions_difference"].mean(),
            "density": density_result.dataset_level.value,
            "coverage": coverage_result.dataset_level.value,
            "coherence": coherence_result.dataset_level.value,
            "authenticity": authenticity_result.dataset_level.value
        })

        print(f"\n--- MODEL: {model} | NOISE: {noise} ---")
        print(numerical_results)
        print(categorical_results)
        print(similarity_result)
        print(divergence_result)
        print(density_result)
        print(coverage_result)
        print(coherence_result)
        print(authenticity_result)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("second_dataset_statistical_metrics_results.csv", index=False)