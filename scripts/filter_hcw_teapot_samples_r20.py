import pandas as pd
from pathlib import Path
import random

# ================= CONFIG =================
INPUT_DIR = Path("outputs/tables")
OUTPUT_DIR = Path("outputs/tables/hcw_teapot_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESH = 0.9
MAX_SAMPLES = 20
RANDOM_SEED = 42
TARGET_CLASS = "teapot"   # ⚠️ 如果你用 idx，这里改成数字

random.seed(RANDOM_SEED)

# ================= LOAD =================
df_A = pd.read_csv(INPUT_DIR / "difficulty_A_test_full.csv")
df_B = pd.read_csv(INPUT_DIR / "difficulty_B_test_full.csv")
df_C = pd.read_csv(INPUT_DIR / "difficulty_C_test_full.csv")
df_D = pd.read_csv(INPUT_DIR / "difficulty_D_test_full.csv")

# ================= HCW FLAG =================
def add_hcw(df):
    df = df.copy()
    df["HCW"] = (df["confidence"] > CONF_THRESH) & (df["is_correct"] == 0)
    return df

df_A = add_hcw(df_A)
df_B = add_hcw(df_B)
df_C = add_hcw(df_C)
df_D = add_hcw(df_D)

# ================= SAMPLING FUNCTION =================
def sample_teapot_hcw(df, model_name):
    """
    Filter teapot HCW samples and randomly select up to MAX_SAMPLES.
    """
    # --- 如果你用 label name ---
    subset = df[
        (df["true_label"] == TARGET_CLASS) &
        (df["HCW"])
    ]

    # --- 如果你用 label index，用这个替换上面 ---
    # subset = df[
    #     (df["true_label_idx"] == TARGET_CLASS) &
    #     (df["HCW"])
    # ]

    print(f"{model_name} teapot HCW total: {len(subset)}")

    if len(subset) > MAX_SAMPLES:
        subset = subset.sample(
            n=MAX_SAMPLES,
            random_state=RANDOM_SEED
        )

    out_path = OUTPUT_DIR / f"{model_name}_teapot_HCW_samples.csv"
    subset.to_csv(out_path, index=False)

    print(f"[Saved] {model_name}: {len(subset)} samples -> {out_path}\n")

# ================= RUN =================
sample_teapot_hcw(df_A, "A")
sample_teapot_hcw(df_B, "B")
sample_teapot_hcw(df_C, "C")
sample_teapot_hcw(df_D, "D")
