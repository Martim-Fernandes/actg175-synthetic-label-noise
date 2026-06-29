import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def apply_random_label_noise(df, target_col, noise_rate, random_state=44):
    df_noisy = df.copy()

    n_rows = df_noisy.shape[0]
    n_noisy = int(n_rows * noise_rate)

    noisy_indices = df_noisy.index.to_series().sample(
        n=n_noisy,
        random_state=random_state
    )

    df_noisy.loc[noisy_indices, target_col] = 1 - df_noisy.loc[noisy_indices, target_col]

    return df_noisy

def save_dataset(df, path):
    df.to_csv(path, index=False)