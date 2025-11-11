import pandas as pd
import numpy as np
import pickle
import json
import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# --- Configuration and File Paths ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT_DIR / "outputs"

# Path to the experiment output directory
EXPERIMENT_NAME = "three_tower_synthetic_kfold_retrieval"
EXPERIMENT_OUTPUT_DIR = OUTPUTS_DIR / EXPERIMENT_NAME

# File paths for input data
EMBEDDINGS_FILE = EXPERIMENT_OUTPUT_DIR / "fold_4_embeddings.pkl"
METRICS_FILE = (
    EXPERIMENT_OUTPUT_DIR / "metrics" / "kfold_retrieval_three_tower_metrics.json"
)
NAHUATL_NAMES_PATH = (
    PROJECT_ROOT_DIR / "data" / "FOR NAHUATL REVIEW - Nahuatl names.csv"
)

# Output directory for final files
FINAL_OUTPUTS_DIR = OUTPUTS_DIR / "figures"
os.makedirs(FINAL_OUTPUTS_DIR, exist_ok=True)

# --- Part 1: Generate and Save the Final Visualization ---
print("--- Part 1: Generating Final Visualization ---")

# --- Load original metadata to get plant names and types ---
if not NAHUATL_NAMES_PATH.exists():
    print(f"Error: Nahuatl names file not found at {NAHUATL_NAMES_PATH}.")
    exit()
nahuatl_df = pd.read_csv(NAHUATL_NAMES_PATH)
nahuatl_df = nahuatl_df[["official_name", "label"]]
nahuatl_df["official_name"] = nahuatl_df["official_name"].str.strip().str.lower()
name_to_type = nahuatl_df.set_index("official_name")["label"].to_dict()

if not EMBEDDINGS_FILE.exists():
    print(f"Error: Embeddings file not found at {EMBEDDINGS_FILE}.")
    print("Please ensure the modified script 17 has been run to save the embeddings.")
    exit()

print(f"Loading embeddings from: {EMBEDDINGS_FILE}")
with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings_data = pickle.load(f)

img_embeddings = embeddings_data["image_embeddings"]
txt_embeddings = embeddings_data["text_embeddings"]
graph_embeddings = embeddings_data["graph_embeddings"]
labels = embeddings_data["labels"]

print(f"Loaded embeddings for {len(img_embeddings)} samples.")

unique_names = sorted(list(nahuatl_df["official_name"].unique()))
label_to_name = {i: name for i, name in enumerate(unique_names)}

all_embeddings = np.concatenate(
    [txt_embeddings, img_embeddings, graph_embeddings], axis=0
)
reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(all_embeddings)

df_umap = pd.DataFrame(embedding_2d, columns=["umap_x", "umap_y"])
df_umap["modality"] = (
    ["Text"] * len(txt_embeddings)
    + ["Image"] * len(img_embeddings)
    + ["Graph"] * len(graph_embeddings)
)
df_umap["label_id"] = np.concatenate([labels, labels, labels], axis=0)
df_umap["official_name"] = df_umap["label_id"].map(label_to_name)
df_umap["type"] = df_umap["official_name"].map(name_to_type)

txt_umap = embedding_2d[: len(txt_embeddings)]
img_umap = embedding_2d[len(txt_embeddings) : 2 * len(txt_embeddings)]
graph_umap = embedding_2d[2 * len(txt_embeddings) :]

distances = []
for i in range(len(img_embeddings)):
    dist_img_txt = np.linalg.norm(img_umap[i] - txt_umap[i])
    dist_img_graph = np.linalg.norm(img_umap[i] - graph_umap[i])
    dist_txt_graph = np.linalg.norm(txt_umap[i] - graph_umap[i])
    name = label_to_name[labels[i]]
    distances.append({"name": name, "pair": "Image-Text", "distance": dist_img_txt})
    distances.append({"name": name, "pair": "Image-Graph", "distance": dist_img_graph})
    distances.append({"name": name, "pair": "Text-Graph", "distance": dist_txt_graph})
df_distances = pd.DataFrame(distances)

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.5))
plt.subplots_adjust(hspace=0.3, wspace=0.2)
xlim = (df_umap["umap_x"].min(), df_umap["umap_x"].max())
ylim = (df_umap["umap_y"].min(), df_umap["umap_y"].max())
highlight_color = "black"
background_color = "lightgray"
highlight_size = 35
background_size = 30
palette_text = {
    "Text": highlight_color,
    "Image": background_color,
    "Graph": background_color,
}
palette_image = {
    "Text": background_color,
    "Image": highlight_color,
    "Graph": background_color,
}
palette_graph = {
    "Text": background_color,
    "Image": background_color,
    "Graph": highlight_color,
}

df_text_highlight = df_umap.sort_values(
    by="modality", key=lambda x: x.map({"Image": 0, "Graph": 1, "Text": 2})
)
sns.scatterplot(
    x="umap_x",
    y="umap_y",
    data=df_text_highlight,
    ax=axes[0, 0],
    edgecolor="none",
    hue="modality",
    palette=palette_text,
    legend=False,
    size="modality",
    sizes={"Text": highlight_size, "Image": background_size, "Graph": background_size},
    alpha=0.7,
)
axes[0, 0].set_xlabel("UMAP Dimension 1", fontsize=10)
axes[0, 0].set_ylabel("UMAP Dimension 2", fontsize=10)
axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)
axes[0, 0].set_title("Text Embeddings", fontsize=12, loc="left")

df_image_highlight = df_umap.sort_values(
    by="modality", key=lambda x: x.map({"Text": 0, "Graph": 1, "Image": 2})
)
sns.scatterplot(
    x="umap_x",
    y="umap_y",
    data=df_image_highlight,
    ax=axes[0, 1],
    edgecolor="none",
    hue="modality",
    palette=palette_image,
    legend=False,
    size="modality",
    sizes={"Text": background_size, "Image": highlight_size, "Graph": background_size},
    alpha=0.7,
)
axes[0, 1].set_xlabel("UMAP Dimension 1", fontsize=10)
axes[0, 1].set_ylabel("UMAP Dimension 2", fontsize=10)
axes[0, 1].set_xlim(xlim)
axes[0, 1].set_ylim(ylim)
axes[0, 1].set_title("Image Embeddings", fontsize=12, loc="left")

df_graph_highlight = df_umap.sort_values(
    by="modality", key=lambda x: x.map({"Image": 0, "Text": 1, "Graph": 2})
)
sns.scatterplot(
    x="umap_x",
    y="umap_y",
    data=df_graph_highlight,
    ax=axes[1, 0],
    edgecolor="none",
    hue="modality",
    palette=palette_graph,
    legend=False,
    size="modality",
    sizes={"Text": background_size, "Image": background_size, "Graph": highlight_size},
    alpha=0.7,
)
axes[1, 0].set_xlabel("UMAP Dimension 1", fontsize=10)
axes[1, 0].set_ylabel("UMAP Dimension 2", fontsize=10)
axes[1, 0].set_xlim(xlim)
axes[1, 0].set_ylim(ylim)
axes[1, 0].set_title("Graph Embeddings", fontsize=12, loc="left")

sns.histplot(
    data=df_distances,
    x="distance",
    hue="pair",
    multiple="stack",
    bins=50,
    ax=axes[1, 1],
    palette={
        "Image-Text": "#1b9e77",
        "Image-Graph": "#d95f02",
        "Text-Graph": "#7570b3",
    },
    legend=False,
)
axes[1, 1].set_xlabel("Euclidean Distance", fontsize=10)
axes[1, 1].set_ylabel("Count", fontsize=10)
axes[1, 1].set_title("Pairwise UMAP Distances", fontsize=12, loc="left")
legend_patches = [
    Patch(color="#1b9e77", label="Image-Text"),
    Patch(color="#d95f02", label="Image-Graph"),
    Patch(color="#7570b3", label="Text-Graph"),
]
axes[1, 1].legend(handles=legend_patches, loc="upper right")

plt.tight_layout()
viz_path = FINAL_OUTPUTS_DIR / "fig_three_towers.png"
plt.savefig(viz_path, bbox_inches="tight", dpi=300)
print(f"Final visualization saved to: {viz_path}")

# --- Part 2: Load and Save the Final Metrics Tables ---
print("\n--- Part 2: Creating and Saving Final Metrics Tables ---")

if not METRICS_FILE.exists():
    print(f"Error: Metrics file not found at {METRICS_FILE}.")
    print("Please ensure the modified script 17 has been run to save the metrics.")
    exit()

with open(METRICS_FILE, "r") as f:
    metrics_data = json.load(f)

# Convert metrics to a pandas DataFrame
all_rows = []
for retrieval_type, data in metrics_data.items():
    row = {
        "Retrieval Type": retrieval_type.replace("_", "-").title().replace("To", "to")
    }

    metrics_to_keep = [
        "mAP_avg",
        "mAP_std",
        "Recall@1_score_avg",
        "Recall@1_score_std",
        "Recall@1_above_chance_avg",
        "Recall@5_score_avg",
        "Recall@5_score_std",
        "Recall@5_above_chance_avg",
        "Recall@10_score_avg",
        "Recall@10_score_std",
        "Recall@10_above_chance_avg",
    ]

    for key in metrics_to_keep:
        value = data.get(key, np.nan)
        if key.endswith("_avg"):
            col_name = (
                key.replace("_avg", " avg.").replace("_", " ").replace("mAP", "MAP")
            )
        elif key.endswith("_std"):
            col_name = (
                key.replace("_std", " std.").replace("_", " ").replace("mAP", "MAP")
            )
        else:
            col_name = (
                key.replace("_", " ")
                .replace("avg", "avg.")
                .replace("chance", "chance.")
            )
        row[col_name] = f"{value:.4f}"

    all_rows.append(row)

df_metrics = pd.DataFrame(all_rows)
df_metrics = df_metrics.set_index("Retrieval Type")

# Transpose the DataFrame
df_metrics_transposed = df_metrics.T
df_metrics_transposed.index.name = "Metric"

# Save to CSV
csv_path = FINAL_OUTPUTS_DIR / "table_three_towers.csv"
df_metrics_transposed.to_csv(csv_path)
print(f"Metrics table saved to CSV: {csv_path}")

# Save to Markdown
md_path = FINAL_OUTPUTS_DIR / "table_three_towers.txt"
markdown_string = df_metrics_transposed.to_markdown()
with open(md_path, "w") as f:
    f.write(markdown_string)
print(f"Metrics table saved to Markdown: {md_path}")

print("\nAll final outputs have been generated successfully.")
