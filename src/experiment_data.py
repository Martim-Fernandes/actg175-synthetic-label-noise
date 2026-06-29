import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("second_dataset_clean.csv")

target_col = "cid"
random_state = 44
test_size = 0.30

train_df, test_df = train_test_split(
    df,
    test_size=test_size,
    random_state=random_state,
    stratify=df[target_col]
)

train_df.to_csv("second_dataset_train.csv", index=False)
test_df.to_csv("second_dataset_test.csv", index=False)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("\nTrain target distribution:")
print(train_df[target_col].value_counts())
print(train_df[target_col].value_counts(normalize=True) * 100)

print("\nTest target distribution:")
print(test_df[target_col].value_counts())
print(test_df[target_col].value_counts(normalize=True) * 100)

print("\nSaved: second_dataset_train.csv")
print("Saved: second_dataset_test.csv")