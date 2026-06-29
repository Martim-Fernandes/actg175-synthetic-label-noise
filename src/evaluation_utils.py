import pandas as pd

def compare_target_distribution(real_df, synthetic_df, target_col):
    print(real_df[target_col].value_counts())
    print(synthetic_df[target_col].value_counts())

    print(real_df[target_col].value_counts(normalize=True) * 100)
    print(synthetic_df[target_col].value_counts(normalize=True) * 100)

def describe_numerical(real_df, synthetic_df, numerical_cols):
    print(real_df[numerical_cols].describe())
    print(synthetic_df[numerical_cols].describe())

def compare_categorical_counts(real_df, synthetic_df, column):
    print(real_df[column].value_counts())
    print(synthetic_df[column].value_counts())

def compare_crosstab(real_df, synthetic_df, feature_col, target_col):
    print(pd.crosstab(real_df[feature_col], real_df[target_col], normalize="index") * 100)
    print(pd.crosstab(synthetic_df[feature_col], synthetic_df[target_col], normalize="index") * 100)