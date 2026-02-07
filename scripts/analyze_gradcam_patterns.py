import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ================= CONFIG =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_DIR = PROJECT_ROOT / "outputs/annotations/gradcam_teapot"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
STATS_DIR = PROJECT_ROOT / "outputs/stats"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "A": "A_teapot_HCW_annotated.csv",
    "B": "B_teapot_HCW_annotated.csv",
    "C": "C_teapot_HCW_annotated.csv",
    "D": "D_teapot_HCW_annotated.csv",
}

PATTERN_ORDER = ["Object", "Edge", "Background", "Wrong-object"]

# ================= LOAD & AGGREGATE =================
summary_rows = []

for model_name, filename in MODEL_FILES.items():
    csv_path = ANNOTATION_DIR / filename
    assert csv_path.exists(), f"File not found: {csv_path}"

    df = pd.read_csv(csv_path)

    total = len(df)

    counts = (
        df["dominant_pattern"]
        .value_counts()
        .reindex(PATTERN_ORDER, fill_value=0)
    )

    for pattern in PATTERN_ORDER:
        count = counts[pattern]
        percentage = count / total * 100

        summary_rows.append({
            "model": model_name,
            "pattern": pattern,
            "count": count,
            "percentage": percentage
        })

summary_df = pd.DataFrame(summary_rows)

# ================= SAVE SUMMARY CSV =================
summary_csv_path = STATS_DIR / "gradcam_teapot_pattern_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)
print(f"[Saved] {summary_csv_path}")

# ================= PLOT (STACKED BAR) =================
pivot_df = summary_df.pivot(
    index="model",
    columns="pattern",
    values="percentage"
).reindex(columns=PATTERN_ORDER)

ax = pivot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5),
    colormap="tab10"
)

ax.set_ylabel("Percentage of HCW samples (%)")
ax.set_xlabel("Model")
ax.set_title("Distribution of Dominant Grad-CAM Failure Patterns (Teapot HCW)")
ax.legend(
    title="Dominant Pattern",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

plt.tight_layout()

fig_path = FIGURE_DIR / "gradcam_teapot_pattern_distribution.png"
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"[Saved] {fig_path}")
