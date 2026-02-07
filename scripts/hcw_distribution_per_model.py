import pandas as pd
from pathlib import Path

# ================= CONFIG =================
INPUT_DIR = Path("outputs/tables")
OUTPUT_DIR = Path("outputs/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESH = 0.9

FILES = {
    "A": "difficulty_A_test_full.csv",
    "B": "difficulty_B_test_full.csv",
    "C": "difficulty_C_test_full.csv",
    "D": "difficulty_D_test_full.csv",
}

# ================= CORE =================
def hcw_distribution(csv_path: Path):
    df = pd.read_csv(csv_path)

    # sanity check（强烈建议你第一次跑时保留）
    print(f"{csv_path.name}: total samples = {len(df)}")

    # 高置信错误：conf > 0.9 且预测错误
    hcw = df[(df["confidence"] > CONF_THRESH) & (df["is_correct"] == 0)]

    dist = (
        hcw.groupby("true_label")
           .size()
           .reset_index(name="HCW_count")
           .sort_values("HCW_count", ascending=False)
    )

    return dist


# ================= RUN =================
for model_id, fname in FILES.items():
    csv_path = INPUT_DIR / fname
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    print(f"\n=== Model {model_id} ===")
    dist = hcw_distribution(csv_path)
    print(dist)

    out_csv = OUTPUT_DIR / f"hcw_distribution_model_{model_id}.csv"
    dist.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
