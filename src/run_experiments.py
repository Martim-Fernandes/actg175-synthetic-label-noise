from data_utils import load_dataset, apply_random_label_noise, save_dataset
from synthetic_utils import train_gaussian_copula, generate_synthetic_data, train_ctgan, train_tvae

dataset_path = "second_dataset_train.csv"
target_col = "cid"
random_state = 44
noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
models = ["GCS", "CTGAN", "TVAE"]

numerical_cols = [
    "age", "wtkg", "karnof", "cd40", "cd420", "cd80", "cd820"
]

categorical_cols = [
    "trt", "hemo", "homo", "drugs", "oprior",
    "z30", "zprior", "preanti", "race", "gender",
    "str2", "strat", "symptom", "treat", "offtrt"
]

df = load_dataset(dataset_path)

for model in models:
    for noise_rate in noise_levels:
        noise_name = int(noise_rate * 100)

        noisy_output_path = f"second_dataset_train_noisy_{noise_name}.csv"
        synthetic_output_path = f"second_dataset_synthetic_noisy_{model}_{noise_name}.csv"

        df_noisy = apply_random_label_noise(
            df,
            target_col,
            noise_rate,
            random_state=random_state
        )

        save_dataset(df_noisy, noisy_output_path)

        if model == "CTGAN":
            synthesizer = train_ctgan(df_noisy)
        elif model == "GCS":
            synthesizer = train_gaussian_copula(df_noisy)
        elif model == "TVAE":
            synthesizer = train_tvae(df_noisy)

        synthetic_df = generate_synthetic_data(synthesizer, df_noisy.shape[0])
        save_dataset(synthetic_df, synthetic_output_path)

        print(f"\n--- MODEL: {model} | NOISE: {noise_name} ---")
        print("Saved:", noisy_output_path)
        print("Saved:", synthetic_output_path)