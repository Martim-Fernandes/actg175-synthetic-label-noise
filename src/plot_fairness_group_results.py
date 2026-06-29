import pandas as pd
import matplotlib.pyplot as plt

results_df = pd.read_csv("second_dataset_fairness_results.csv")

metrics_to_plot = ["accuracy", "f1", "roc_auc"]
group_columns = results_df["group_column"].unique()
classifiers = results_df["classifier"].unique()
generator_models = results_df["generator_model"].unique()

for group_col in group_columns:
    group_df = results_df[results_df["group_column"] == group_col]

    for classifier in classifiers:
        classifier_df = group_df[group_df["classifier"] == classifier]

        for generator_model in generator_models:
            model_df = classifier_df[classifier_df["generator_model"] == generator_model]

            for metric in metrics_to_plot:
                plt.figure()

                for group_value in model_df["group_value"].unique():
                    value_df = model_df[
                        model_df["group_value"] == group_value
                    ].sort_values("noise")

                    plt.plot(
                        value_df["noise"],
                        value_df[metric],
                        marker="o",
                        label=group_value
                    )

                plt.title(f"{group_col} - {classifier} - {generator_model} - {metric}")
                plt.xlabel("Noise level (%)")
                plt.ylabel(metric)
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    f"second_dataset_{group_col}_{classifier}_{generator_model}_{metric}.png",
                    bbox_inches="tight"
                )
                plt.show()