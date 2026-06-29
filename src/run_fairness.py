import pandas as pd
from sklearn.model_selection import train_test_split

from classification_utils import (
    prepare_features_target,
    encode_train_test,
    train_decision_tree,
    train_random_forest,
    train_logistic_regression
)

from fairness_utils import (
    evaluate_by_group,
    summarize_group_gap
)

target_col = "cid"
random_state = 44
test_size = 0.30
split_name = "70/30"

generator_models = ["GCS", "CTGAN", "TVAE"]
classifiers = ["DecisionTree", "RandomForest", "LogisticRegression"]
noise_levels = [0, 10, 20, 30, 40, 50]
group_columns = ["gender", "race"]

results = []
gap_results = []

for noise in noise_levels:
    print(f"\n--- NOISE: {noise} ---")

    if noise == 0:
        real_df = pd.read_csv("second_dataset_train.csv")
    else:
        real_df = pd.read_csv(f"second_dataset_train_noisy_{noise}.csv")

    real_train, real_test = train_test_split(
        real_df,
        test_size=test_size,
        random_state=random_state,
        stratify=real_df[target_col]
    )

    X_train_real, y_train_real = prepare_features_target(real_train, target_col)
    X_test_real, y_test_real = prepare_features_target(real_test, target_col)

    X_test_groups = X_test_real.copy()

    X_train_real_enc, X_test_real_enc = encode_train_test(X_train_real, X_test_real)

    for classifier in classifiers:
        print(f"Running real_baseline | {classifier}")

        if classifier == "DecisionTree":
            model_real = train_decision_tree(X_train_real_enc, y_train_real, random_state=random_state)
        elif classifier == "RandomForest":
            model_real = train_random_forest(X_train_real_enc, y_train_real, random_state=random_state)
        elif classifier == "LogisticRegression":
            model_real = train_logistic_regression(X_train_real_enc, y_train_real, random_state=random_state)

        y_pred_real = model_real.predict(X_test_real_enc)
        y_proba_real = model_real.predict_proba(X_test_real_enc)[:, 1]

        for group_col in group_columns:
            group_results = evaluate_by_group(
                y_true=y_test_real,
                y_pred=y_pred_real,
                y_proba=y_proba_real,
                group_series=X_test_groups[group_col],
                min_samples=30
            )

            group_results["group_column"] = group_col
            group_results["generator_model"] = "real_baseline"
            group_results["classifier"] = classifier
            group_results["noise"] = noise
            group_results["split"] = split_name

            results.append(group_results)

            gap_summary = summarize_group_gap(group_results, group_col)
            gap_summary["generator_model"] = "real_baseline"
            gap_summary["classifier"] = classifier
            gap_summary["noise"] = noise
            gap_summary["split"] = split_name

            gap_results.append(gap_summary)

    for generator_model in generator_models:
        print(f"\nGenerator: {generator_model}")

        if generator_model == "GCS":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_GCS_{noise}.csv")
        elif generator_model == "CTGAN":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_CTGAN_{noise}.csv")
        elif generator_model == "TVAE":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_TVAE_{noise}.csv")

        X_train_syn, y_train_syn = prepare_features_target(synthetic_df, target_col)
        X_test_real, y_test_real = prepare_features_target(real_test, target_col)

        X_test_groups = X_test_real.copy()

        X_train_syn_enc, X_test_real_enc = encode_train_test(X_train_syn, X_test_real)

        for classifier in classifiers:
            print(f"Running {generator_model} | {classifier}")

            if classifier == "DecisionTree":
                model_syn = train_decision_tree(X_train_syn_enc, y_train_syn, random_state=random_state)
            elif classifier == "RandomForest":
                model_syn = train_random_forest(X_train_syn_enc, y_train_syn, random_state=random_state)
            elif classifier == "LogisticRegression":
                model_syn = train_logistic_regression(X_train_syn_enc, y_train_syn, random_state=random_state)

            y_pred_syn = model_syn.predict(X_test_real_enc)
            y_proba_syn = model_syn.predict_proba(X_test_real_enc)[:, 1]

            for group_col in group_columns:
                group_results = evaluate_by_group(
                    y_true=y_test_real,
                    y_pred=y_pred_syn,
                    y_proba=y_proba_syn,
                    group_series=X_test_groups[group_col],
                    min_samples=30
                )

                group_results["group_column"] = group_col
                group_results["generator_model"] = generator_model
                group_results["classifier"] = classifier
                group_results["noise"] = noise
                group_results["split"] = split_name

                results.append(group_results)

                gap_summary = summarize_group_gap(group_results, group_col)
                gap_summary["generator_model"] = generator_model
                gap_summary["classifier"] = classifier
                gap_summary["noise"] = noise
                gap_summary["split"] = split_name

                gap_results.append(gap_summary)

results_df = pd.concat(results, ignore_index=True)
gap_results_df = pd.DataFrame(gap_results)

print(results_df)
print(gap_results_df)

results_df.to_csv("second_dataset_fairness_results.csv", index=False)
gap_results_df.to_csv("second_dataset_fairness_gap_results.csv", index=False)