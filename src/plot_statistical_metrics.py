import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("second_dataset_statistical_metrics_results.csv")

metrics_to_plot = [
    "similarity_mean",
    "divergence_js_mean",
    "numerical_mean_abs_diff",
    "categorical_mean_abs_prop_diff",
    "density",
    "coverage",
    "coherence",
    "authenticity"
]

for metric in metrics_to_plot:
    plt.figure()

    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model].sort_values("noise")
        plt.plot(model_df["noise"], model_df[metric], marker="o", label=model)

    plt.title(f"{metric} by noise level")
    plt.xlabel("Noise level (%)")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"second_dataset_{metric}.png", bbox_inches="tight")
    plt.show()