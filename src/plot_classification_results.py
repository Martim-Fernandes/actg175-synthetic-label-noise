import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("second_dataset_classification_results_70_30.csv")

metrics_to_plot = ["accuracy", "f1", "roc_auc"]
classifiers = results_df["classifier"].unique()

for classifier in classifiers:
    classifier_df = results_df[results_df["classifier"] == classifier]

    for metric in metrics_to_plot:
        plt.figure()

        for generator_model in classifier_df["generator_model"].unique():
            model_df = classifier_df[
                classifier_df["generator_model"] == generator_model
            ].sort_values("noise")

            plt.plot(
                model_df["noise"],
                model_df[metric],
                marker="o",
                label=generator_model
            )

        plt.title(f"{classifier} - {metric} by noise level")
        plt.xlabel("Noise level (%)")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(f"second_dataset_{classifier}_{metric}.png", bbox_inches="tight")
        plt.show()