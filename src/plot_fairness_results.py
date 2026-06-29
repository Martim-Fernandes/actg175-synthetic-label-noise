import pandas as pd
import matplotlib.pyplot as plt

gap_df = pd.read_csv("second_dataset_fairness_gap_results.csv")

metrics_to_plot = ["accuracy_gap", "f1_gap", "roc_auc_gap"]
group_columns = gap_df["group_column"].unique()
classifiers = gap_df["classifier"].unique()

for group_col in group_columns:
    group_df = gap_df[gap_df["group_column"] == group_col]

    for classifier in classifiers:
        classifier_df = group_df[group_df["classifier"] == classifier]

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

            plt.title(f"{group_col} - {classifier} - {metric}")
            plt.xlabel("Noise level (%)")
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.savefig(f"second_dataset_{group_col}_{classifier}_{metric}.png", bbox_inches="tight")
            plt.show()