import pandas as pd
from ucimlrepo import fetch_ucirepo

actg175 = fetch_ucirepo(id=890)

X = actg175.data.features
y = actg175.data.targets

df = pd.concat([X, y], axis=1)

target_col = "cid"

selected_cols = [
    "trt", "age", "wtkg", "hemo", "homo", "drugs", "karnof",
    "oprior", "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "treat", "offtrt",
    "cd40", "cd420", "cd80", "cd820", "cid"
]

df = df[selected_cols]

print(df.shape)
print(df[target_col].value_counts())
print(df[target_col].value_counts(normalize=True) * 100)

df.to_csv("second_dataset_clean.csv", index=False)