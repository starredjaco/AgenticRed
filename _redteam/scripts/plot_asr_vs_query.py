import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# Global Plot Style (NeurIPS/ICLR-ready)
# ---------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "font.size": 15,
    "font.family": "serif",
    "axes.labelsize": "large",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "legend.frameon": False,
    'xtick.labelsize': 10
})

csv_path = "/NS/MAS-llms02/work/jyuan/AgenticRed/_redteam/scripts/csv_assets/asr_query.csv"
out_path = "/NS/MAS-llms02/work/jyuan/AgenticRed/_redteam/scripts/csv_assets/asr_vs_query_log_husl.png"

# Seaborn style
# sns.set_theme(style="whitegrid")

df = pd.read_csv(csv_path)


# Wide -> long: one point per (family, model, target)
rows = []
for _, r in df.iterrows():
    rows.append({
        "family": r["family"],
        "model": r["model"],
        "target": "Qwen3",
        "asr": r["qwen3_asr"],
        "query_time": r["qwen3_query_time"],
    })
    rows.append({
        "family": r["family"],
        "model": r["model"],
        "target": "Llama2",
        "asr": r["llama2_asr"],
        "query_time": r["llama2_query_time"],
    })

long_df = pd.DataFrame(rows)
long_df["asr"] = pd.to_numeric(long_df["asr"], errors="coerce")
long_df["query_time"] = pd.to_numeric(long_df["query_time"], errors="coerce")
long_df = long_df.dropna(subset=["asr", "query_time"])

# Color by family using husl
families = sorted(long_df["family"].unique())
fam_palette = sns.color_palette("husl", n_colors=len(families))
family_color = {f: fam_palette[i] for i, f in enumerate(families)}

# Shape by model (different models, including within the same family)
models = sorted(long_df["model"].unique())
model_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
model_marker = {m: model_markers[i % len(model_markers)] for i, m in enumerate(models)}

# Fill style by target model: filled for Qwen3, hollow for Llama2
target_filled = {"Qwen3": True, "Llama2": False}

fig, ax = plt.subplots(figsize=(9, 6))
# Leave a dedicated right margin for legends
fig.subplots_adjust(right=0.72)

for _, r in long_df.iterrows():
    is_filled = target_filled[r["target"]]
    face = family_color[r["family"]] if is_filled else "none"
    ax.scatter(
        r["query_time"],
        r["asr"],
        facecolors=face,
        edgecolors=family_color[r["family"]],
        marker=model_marker[r["model"]],
        s=120,
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )

# Required: log scale on query time
ax.set_xscale("log")
ax.set_xlabel("Query per Success (log scale)")
ax.set_ylabel("ASR")
ax.set_title("ASR vs Query Time by Across Target Models", fontsize=16, weight='bold')
ax.set_ylim(0, 105)

# Legend 1: family colors
family_handles = [
    Line2D([0], [0], marker="o", linestyle="", markerfacecolor=family_color[f],
           markeredgecolor="black", markersize=8, label=f)
    for f in families
]

# Legend 2: model shapes
model_handles = [
    Line2D(
        [0],
        [0],
        marker=model_marker[m],
        linestyle="",
        color="black",
        markerfacecolor="lightgray",
        markeredgecolor="black",
        markersize=8,
        label=m,
    )
    for m in models
]

# Legend 3: target fill styles
target_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        color="black",
        markerfacecolor=("black" if target_filled[t] else "none"),
        markeredgecolor="black",
        markersize=8,
        label=t,
    )
    for t in ["Qwen3", "Llama2"]
]


leg1 = fig.legend(
    handles=family_handles,
    title="Model Family",
    title_fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0.74, 0.88),
    frameon=True,
    fontsize=9,
    borderaxespad=0.0,
)

leg2 = fig.legend(
    handles=model_handles,
    title="Target Model",
    title_fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0.74, 0.58),
    frameon=True,
    borderaxespad=0.0,
    fontsize=9,
)

leg3 = fig.legend(
    handles=target_handles,
    title="Original Target Model",
    title_fontsize=9,
    loc="upper left",
    bbox_to_anchor=(0.74, 0.23),
    frameon=True,
    borderaxespad=0.0,
    fontsize=9,
)

for leg in [leg1, leg2, leg3]:
    leg._legend_box.align = "left"

plt.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)