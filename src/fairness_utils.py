import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_by_group(y_true, y_pred, y_proba, group_series, min_samples=30):
    results = []

    y_true_binary = pd.Series(y_true, index=group_series.index)
    y_pred_binary = pd.Series(y_pred, index=group_series.index)

    for group_value in group_series.dropna().unique():
        mask = group_series == group_value

        if mask.sum() < min_samples:
            continue

        y_true_group = y_true_binary[mask]
        y_pred_group = y_pred_binary[mask]
        y_proba_group = y_proba[mask]

        if y_true_group.nunique() < 2:
            roc_auc_value = None
        else:
            roc_auc_value = roc_auc_score(y_true_group, y_proba_group)

        results.append({
            "group_value": group_value,
            "n_samples": mask.sum(),
            "accuracy": accuracy_score(y_true_group, y_pred_group),
            "f1": f1_score(y_true_group, y_pred_group),
            "roc_auc": roc_auc_value
        })

    return pd.DataFrame(results)

def summarize_group_gap(group_results, group_column_name):
    return {
        "group_column": group_column_name,
        "accuracy_gap": group_results["accuracy"].max() - group_results["accuracy"].min(),
        "f1_gap": group_results["f1"].max() - group_results["f1"].min(),
        "roc_auc_gap": group_results["roc_auc"].max() - group_results["roc_auc"].min()
    }