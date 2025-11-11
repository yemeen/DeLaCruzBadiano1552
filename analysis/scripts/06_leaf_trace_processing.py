import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math
import re
import urllib.request
import zipfile

# --- Configuration and Path Setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEAF_TRACES_DIR = PROJECT_ROOT / "data" / "leaf_traces"
PICS_DIR = LEAF_TRACES_DIR / "pics"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "processed_leaf_traces"
TRACE_CHECKS_DIR = OUTPUT_DIR / "trace_checks"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_MASTER_SHEET_DIR = PROJECT_ROOT / "outputs" / "master_sheet_processing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRACE_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MASTER_SHEET_DIR.mkdir(parents=True, exist_ok=True)

# Define the folders to process
LEAF_TRACE_FOLDERS = ["ale", "dan", "kylie", "lachlann", "mariana", "noe", "zoe"]
JIMENA_FOLDER = "jimena"

print("--- Starting 07_leaf_trace_processing.py ---")
print(f"Project root is: {PROJECT_ROOT}")


def download_and_extract_images():
    """
    Downloads herbal page images from figshare if not already present.
    """
    if PICS_DIR.exists() and list(PICS_DIR.glob("*.png")):
        print(f"Images already present in {PICS_DIR}. Skipping download.")
        return

    LEAF_TRACES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading herbal page images from figshare...")

    url = "https://figshare.com/ndownloader/files/56900486"
    zip_path = LEAF_TRACES_DIR / "herbal_images.zip"

    try:
        print(f"Downloading to {zip_path}...")

        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 // total_size, 100)
            print(f"\rProgress: {percent}%", end="")

        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\nDownload complete!")

        print(f"Extracting images to {LEAF_TRACES_DIR}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(LEAF_TRACES_DIR)

        import shutil

        macosx_dir = LEAF_TRACES_DIR / "__MACOSX"
        if macosx_dir.exists():
            shutil.rmtree(macosx_dir)

        zip_path.unlink()
        print("Images ready for processing!")

    except Exception as e:
        print(f"\nError downloading or extracting images: {e}")
        raise


download_and_extract_images()

# --- 1. Data Parsing and Collection ---
print("\nProcessing leaf trace data files...")
plant_traces = {}
total_traces = 0

# Process standard folders
for folder_name in LEAF_TRACE_FOLDERS:
    folder_path = LEAF_TRACES_DIR / folder_name
    if not folder_path.exists():
        print(f"Warning: Folder '{folder_name}' not found. Skipping.")
        continue

    for file_path in folder_path.glob("*.txt"):
        plant_id = file_path.stem.rsplit("_", 1)[0]

        # --- FIX FOR MISLABELED ID p055_1 ---
        if plant_id == "p055_1":
            plant_id = "p055_01"

        try:
            coords = np.loadtxt(file_path, delimiter="\t")
            if plant_id not in plant_traces:
                plant_traces[plant_id] = []
            plant_traces[plant_id].append(coords)
            total_traces += 1
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

# Process jimena folder (special case)
jimena_path = LEAF_TRACES_DIR / JIMENA_FOLDER
if jimena_path.exists():
    for subfolder in jimena_path.iterdir():
        if subfolder.is_dir():
            plant_id = subfolder.name
            for file_path in subfolder.glob("*.txt"):
                try:
                    coords = np.loadtxt(file_path, delimiter="\t")
                    if plant_id not in plant_traces:
                        plant_traces[plant_id] = []
                    plant_traces[plant_id].append(coords)
                    total_traces += 1
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
else:
    print(f"Warning: Jimena folder '{JIMENA_FOLDER}' not found. Skipping.")

print(f"\nFound traces for {len(plant_traces)} unique plants.")
print(f"Found a total of {total_traces} individual leaf traces.")
print("--- Data collection complete ---")


# --- 2. Data Audit and Master Sheet Processing ---
print("\n--- 2/5: Processing master sheet and generating data audit report ---")

# File paths
MASTER_SHEET = PROJECT_ROOT / "data" / "FOR NAHUATL REVIEW - Nahuatl names.csv"
VERIFIED_SHEET = OUTPUT_MASTER_SHEET_DIR / "verified_subchapter_links.csv"

# --- A) Get all illustration IDs and metadata from the master sheet ---
if not MASTER_SHEET.exists():
    print(f"Error: Master sheet not found at {MASTER_SHEET}. Cannot perform audit.")
    exit()

master_df = pd.read_csv(MASTER_SHEET)
master_sheet_data = []

for _, row in master_df.iterrows():
    illustrations = str(row.get("illustrations", "")).strip()
    if illustrations and illustrations.lower() != "nan":
        illustration_ids = [pid.strip() for pid in illustrations.split(";")]
        for ill_id in illustration_ids:
            master_sheet_data.append(
                {
                    "illustration_id": ill_id,
                    "ID": row.get("ID"),
                    "official_name": row.get("official_name"),
                    "is_in_master_sheet": True,
                }
            )

master_sheet_df = pd.DataFrame(master_sheet_data)
if not master_sheet_df.empty:
    master_sheet_df = (
        master_sheet_df.groupby("illustration_id")
        .agg(
            {
                "ID": lambda x: "; ".join(x.dropna().unique()),
                "official_name": lambda x: "; ".join(x.dropna().unique()),
                "is_in_master_sheet": "first",
            }
        )
        .reset_index()
    )
master_sheet_ids = set(master_sheet_df["illustration_id"])
print(f"Found {len(master_sheet_ids)} unique illustration IDs in master sheet.")

# --- B) Get all illustration IDs from the verified (text) sheet ---
if VERIFIED_SHEET.exists():
    verified_df = pd.read_csv(VERIFIED_SHEET)
    verified_df["illustrations"] = (
        verified_df["illustrations"].astype(str).str.split(";").str.join("; ")
    )
    verified_ids = set(
        verified_df["illustrations"]
        .str.strip()
        .str.split("; ")
        .explode()
        .dropna()
        .unique()
    )
else:
    verified_ids = set()
    print(
        f"Warning: Verified sheet not found at {VERIFIED_SHEET}. Assuming no plant IDs found in text."
    )

print(f"Found {len(verified_ids)} unique illustration IDs in verified text sheet.")

# --- C) Get all illustration IDs from the processed leaf traces ---
trace_ids = set(plant_traces.keys())
print(f"Found {len(trace_ids)} unique illustration IDs from leaf traces.")

# --- D) Create a final data audit report ---
all_unique_ids = master_sheet_ids.union(verified_ids).union(trace_ids)
report_data = []

for ill_id in sorted(list(all_unique_ids)):
    is_in_master = ill_id in master_sheet_ids
    is_in_verified = ill_id in verified_ids
    is_in_trace = ill_id in trace_ids

    # Find the corresponding metadata from the master sheet data
    metadata_row = master_sheet_df[master_sheet_df["illustration_id"] == ill_id]
    official_name = (
        metadata_row["official_name"].iloc[0] if not metadata_row.empty else None
    )
    plant_id = metadata_row["ID"].iloc[0] if not metadata_row.empty else None

    report_data.append(
        {
            "illustration_id": ill_id,
            "is_in_master_sheet": is_in_master,
            "is_in_verified_text": is_in_verified,
            "is_in_leaf_traces": is_in_trace,
            "ID": plant_id,
            "official_name": official_name,
        }
    )

audit_report_df = pd.DataFrame(report_data)
audit_report_path = OUTPUT_DIR / "data_audit_report.csv"
audit_report_df.to_csv(audit_report_path, index=False)
print(f"Data audit report saved to: {audit_report_path}")

# --- E) Print Summary Statistics ---
illustrated_and_verified = len(trace_ids.intersection(verified_ids))
illustrated_not_verified = len(trace_ids.difference(verified_ids))
print(f"\nSummary of leaf trace data:")
print(
    f"  - Number of illustrated plants also verified in text: {illustrated_and_verified}"
)
print(
    f"  - Number of illustrated plants NOT verified in text: {illustrated_not_verified}"
)

print("--- Data audit complete ---")


# --- 3. Visualization of Individual "Check" Images ---
print("\nGenerating individual cropped trace check images...")
unique_plant_ids = sorted(list(plant_traces.keys()))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
trace_check_paths = {}

for i, plant_id in enumerate(unique_plant_ids):
    traces = plant_traces[plant_id]

    # Extract page number for image
    page_match = re.match(r"(p\d{3})", plant_id)
    if not page_match:
        print(f"Warning: Could not determine page for plant {plant_id}. Skipping.")
        continue
    page_id = page_match.group(1)
    page_png_path = PICS_DIR / f"{page_id}.png"

    if not page_png_path.exists():
        print(f"Warning: Image '{page_png_path.name}' not found. Skipping.")
        continue

    # Load, convert to grayscale, and invert the image for a black background
    original_image = mpimg.imread(page_png_path)
    if original_image.ndim == 3 and original_image.shape[-1] == 4:
        original_image = original_image[:, :, :3]
    grayscale_image = np.dot(original_image[:, :, :3], [0.2989, 0.5870, 0.1140])
    inverted_image = 1 - (grayscale_image / 255.0)

    # Create figure and plot the inverted image first
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        inverted_image,
        cmap="gray",
        extent=[0, original_image.shape[1], original_image.shape[0], 0],
    )
    ax.set_facecolor("black")

    all_x = np.array([])
    all_y = np.array([])

    # Plot the filled leaf traces on top with outlines
    for j, trace in enumerate(traces):
        x, y = trace[:, 0], trace[:, 1]
        color = colors[j % len(colors)]
        ax.fill(x, y, color=color, alpha=1, edgecolor=color, linewidth=0.5)
        all_x = np.concatenate((all_x, x))
        all_y = np.concatenate((all_y, y))

    # Calculate bounding box for cropping with padding
    if all_x.size > 0 and all_y.size > 0:
        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()
        padding = 10
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(max_y + padding, min_y - padding)

    # Clean up the plot
    ax.axis("off")

    # Save the cropped figure
    output_filename = f"{plant_id}_check.png"
    output_path = TRACE_CHECKS_DIR / output_filename
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    trace_check_paths[plant_id] = output_path

    if (i + 1) % 25 == 0 or (i + 1) == len(unique_plant_ids):
        print(f"  - Generated {i + 1}/{len(unique_plant_ids)} check images...")

print("--- Individual check image generation complete ---")


# --- 4. Generate Two Tiled Figures ---
print("\nGenerating two tiled figures of all traces...")

n_plants = len(unique_plant_ids)
half_point = n_plants // 2
plant_ids_part1 = unique_plant_ids[:half_point]
plant_ids_part2 = unique_plant_ids[half_point:]


def generate_tiled_figure(plant_ids, output_filename):
    n_p = len(plant_ids)
    if n_p == 0:
        print(f"Warning: No plants to tile for {output_filename}")
        return

    n_cols = int(math.ceil(math.sqrt(n_p)))
    n_rows = int(math.ceil(n_p / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5, 11), dpi=150)
    axes = axes.flatten()
    fig.set_facecolor("black")

    for i, plant_id in enumerate(plant_ids):
        img_path = trace_check_paths.get(plant_id)
        if img_path and img_path.exists():
            img = mpimg.imread(img_path)
            axes[i].imshow(img)
            axes[i].axis("off")
        else:
            axes[i].axis("off")
            print(f"Warning: Check image not found for {plant_id}")

    for j in range(n_p, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(pad=0.1)
    filepath = OUTPUT_FIGURES_DIR / output_filename
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Tiled figure saved to: {filepath}")


generate_tiled_figure(plant_ids_part1, "leaf_traces_1.png")
generate_tiled_figure(plant_ids_part2, "leaf_traces_2.png")

print("\n--- Tiled figure generation complete ---")
print(f"--- 07_leaf_trace_processing.py Finished ---")
