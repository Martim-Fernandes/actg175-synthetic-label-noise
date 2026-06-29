import pandas as pd
from sklearn.model_selection import train_test_split

from classification_utils import (
    prepare_features_target,
    encode_train_test,
    train_decision_tree,
    train_random_forest,
    train_logistic_regression,
    evaluate_classifier
)

target_col = "cid"
random_state = 44
test_size = 0.30
split_name = "70/30"

models = ["GCS", "CTGAN", "TVAE"]
noise_levels = [0, 10, 20, 30, 40, 50]
classifiers = ["DecisionTree", "RandomForest", "LogisticRegression"]

results = []

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

    X_train_real_enc, X_test_real_enc = encode_train_test(X_train_real, X_test_real)

    for classifier in classifiers:
        print(f"Running real_baseline | {classifier}")

        if classifier == "DecisionTree":
            model_real = train_decision_tree(X_train_real_enc, y_train_real, random_state=random_state)
        elif classifier == "RandomForest":
            model_real = train_random_forest(X_train_real_enc, y_train_real, random_state=random_state)
        elif classifier == "LogisticRegression":
            model_real = train_logistic_regression(X_train_real_enc, y_train_real, random_state=random_state)

        metrics_real = evaluate_classifier(model_real, X_test_real_enc, y_test_real)

        results.append({
            "train_source": "real",
            "generator_model": "real_baseline",
            "classifier": classifier,
            "noise": noise,
            "split": split_name,
            "accuracy": metrics_real["accuracy"],
            "f1": metrics_real["f1"],
            "roc_auc": metrics_real["roc_auc"]
        })

    for model in models:
        if model == "CTGAN":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_CTGAN_{noise}.csv")
        elif model == "GCS":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_GCS_{noise}.csv")
        elif model == "TVAE":
            synthetic_df = pd.read_csv(f"second_dataset_synthetic_noisy_TVAE_{noise}.csv")

        X_train_syn, y_train_syn = prepare_features_target(synthetic_df, target_col)
        X_test_real, y_test_real = prepare_features_target(real_test, target_col)

        X_train_syn_enc, X_test_real_enc = encode_train_test(X_train_syn, X_test_real)

        for classifier in classifiers:
            print(f"Running {model} | {classifier}")

            if classifier == "DecisionTree":
                model_syn = train_decision_tree(X_train_syn_enc, y_train_syn, random_state=random_state)
            elif classifier == "RandomForest":
                model_syn = train_random_forest(X_train_syn_enc, y_train_syn, random_state=random_state)
            elif classifier == "LogisticRegression":
                model_syn = train_logistic_regression(X_train_syn_enc, y_train_syn, random_state=random_state)

            metrics_syn = evaluate_classifier(model_syn, X_test_real_enc, y_test_real)

            results.append({
                "train_source": "synthetic",
                "generator_model": model,
                "classifier": classifier,
                "noise": noise,
                "split": split_name,
                "accuracy": metrics_syn["accuracy"],
                "f1": metrics_syn["f1"],
                "roc_auc": metrics_syn["roc_auc"]
            })

results_df = pd.DataFrame(results)

print(results_df)
results_df.to_csv("second_dataset_classification_results_70_30.csv", index=False)