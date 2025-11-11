#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import shutil
import cv2
import os
import matplotlib.cm as cm
import h5py
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from ect import ECT, EmbeddedGraph


############################
### CONFIGURATION ###
############################

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Input Data Configuration ---
INPUT_DIR = PROJECT_ROOT / "outputs" / "morphometrics"
PCA_PARAMS_FILE = INPUT_DIR / "pca_model_parameters.h5"
PROCESSED_DATA_FILE = INPUT_DIR / "aligned_coords_scores_metadata.h5"

# --- Output Directory Structure for Synthetic Samples ---
SYNTHETIC_DATA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "synthetic_leaf_data"
SYNTHETIC_SHAPE_MASK_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_masks"
SYNTHETIC_SHAPE_ECT_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "shape_ects"
SYNTHETIC_COMBINED_VIZ_DIR = SYNTHETIC_DATA_OUTPUT_DIR / "combined_viz"

SYNTHETIC_METADATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"
FINAL_PREPARED_DATA_FILE = SYNTHETIC_DATA_OUTPUT_DIR / "final_cnn_dataset.pkl"

# --- Shape Information (from previous script's `INTERPOLATION_POINTS`) ---
TOTAL_CONTOUR_COORDS = (
    200
)
FLATTENED_COORD_DIM = TOTAL_CONTOUR_COORDS * 2

# --- ECT (Euler Characteristic Transform) Parameters ---
BOUND_RADIUS = 1
NUM_ECT_DIRECTIONS = 180
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)

# --- Output Image Parameters ---
IMAGE_SIZE = (256, 256)
MASK_BACKGROUND_GRAY = 0
MASK_SHAPE_GRAY = 255

# --- Combined Visualization Parameters ---
OUTLINE_LINE_WIDTH = 2

# --- SMOTE-like Augmentation Parameters ---
K_NEIGHBORS_SMOTE = 5
MIN_SAMPLES_FOR_AUGMENTATION = K_NEIGHBORS_SMOTE

# --- Random Rotation for Data Augmentation ---
APPLY_RANDOM_ROTATION = False
RANDOM_ROTATION_RANGE_DEG = (-180, 180)

# Global ECT min/max will be calculated dynamically
GLOBAL_ECT_MIN = None
GLOBAL_ECT_MAX = None

###########################
### HELPER FUNCTIONS ###
###########################


def apply_transformation_with_affine_matrix(
    points: np.ndarray, affine_matrix: np.ndarray
):
    """
    Applies a 3x3 affine matrix to a set of 2D points (N, 2).
    """
    if points.size == 0:
        return np.array([])

    original_shape = points.shape
    if points.ndim == 1 and points.shape[0] == 2:
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"Input 'points' must be a (N, 2) array or a (2,) array. Got shape: {original_shape}"
        )
    if affine_matrix.shape != (3, 3):
        raise ValueError(
            f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}"
        )

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_homogeneous = points_homogeneous @ affine_matrix.T

    if original_shape == (2,):
        return transformed_homogeneous[0, :2]
    return transformed_homogeneous[:, :2]


def find_robust_affine_transformation_matrix(
    src_points: np.ndarray, dst_points: np.ndarray
):
    """
    Finds a robust affine transformation matrix between source and destination points.
    """
    if len(src_points) < 3 or len(dst_points) < 3:
        if len(src_points) == 0:
            return np.eye(3)
        raise ValueError(
            f"Need at least 3 points to compute affine transformation. Got {len(src_points)}."
        )
    chosen_src_pts = []
    indices = np.arange(len(src_points))
    num_attempts = min(
        len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000
    )
    for _ in range(num_attempts):
        selected_indices = np.random.choice(indices, 3, replace=False)
        p1_src, p2_src, p3_src = src_points[selected_indices]
        area_val = (
            p1_src[0] * (p2_src[1] - p3_src[1])
            + p2_src[0] * (p3_src[1] - p1_src[1])
            + p3_src[0] * (p1_src[1] - p2_src[1])
        )
        if np.abs(area_val) > 1e-6:
            chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
            chosen_dst_pts = np.float32([dst_points[i] for i in selected_indices])
            break
    if len(chosen_src_pts) < 3:
        raise ValueError(
            "Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate."
        )
    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    if M_2x3.shape != (2, 3):
        raise ValueError(
            f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}"
        )
    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    return affine_matrix_3x3


def ect_coords_to_pixels(
    coords_ect: np.ndarray, image_size: tuple, bound_radius: float
):
    """
    Transforms coordinates from ECT space to image pixel space.
    """
    if len(coords_ect) == 0:
        return np.array([])
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]
    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2
    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int)
    return np.column_stack((pixel_x, pixel_y))


def save_grayscale_shape_mask(transformed_coords: np.ndarray, save_path: Path):
    """
    Saves a grayscale image representing a transformed contour/pixel set.
    """
    img = Image.new("L", IMAGE_SIZE, MASK_BACKGROUND_GRAY)
    draw = ImageDraw.Draw(img)
    if transformed_coords is not None and transformed_coords.size > 0:
        pixel_coords = ect_coords_to_pixels(
            transformed_coords, IMAGE_SIZE, BOUND_RADIUS
        )
        pixel_coords = np.clip(
            pixel_coords, [0, 0], [IMAGE_SIZE[0] - 1, IMAGE_SIZE[1] - 1]
        )
        if len(pixel_coords) >= 3:
            polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
            draw.polygon(polygon_points, fill=MASK_SHAPE_GRAY)
        elif len(pixel_coords) > 0:
            for x, y in pixel_coords:
                draw.point((x, y), fill=MASK_SHAPE_GRAY)
    img.save(save_path)


def save_radial_ect_image(
    ect_result,
    save_path: Path,
    cmap_name: str = "gray",
    vmin: float = None,
    vmax: float = None,
):
    """
    Saves the radial ECT plot as an image with the specified colormap.
    """
    if ect_result is None:
        Image.new("L", IMAGE_SIZE, 0).save(save_path)
        return
    fig, ax = plt.subplots(
        subplot_kw=dict(projection="polar"),
        figsize=(IMAGE_SIZE[0] / 100, IMAGE_SIZE[1] / 100),
        dpi=100,
        facecolor="white",
    )
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    im = ax.pcolormesh(THETA, R, ect_result.T, cmap=cmap_name, vmin=vmin, vmax=vmax)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=100,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def create_combined_viz_from_images(
    ect_image_path: Path,
    overlay_coords: np.ndarray,
    save_path: Path,
    overlay_color: tuple,
    overlay_alpha: float,
    overlay_type: str = "points",
    line_width: int = 1,
):
    """
    Creates a combined visualization by overlaying transformed elements (e.g., leaf shape)
    onto the ECT image.
    """
    try:
        ect_img = Image.open(ect_image_path).convert("RGBA")
        img_width, img_height = ect_img.size
        composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
        draw_composite = ImageDraw.Draw(composite_overlay)
        if overlay_coords is not None and overlay_coords.size > 0:
            pixel_coords = ect_coords_to_pixels(
                overlay_coords, IMAGE_SIZE, BOUND_RADIUS
            )
            pixel_coords = np.clip(
                pixel_coords, [0, 0], [img_width - 1, img_height - 1]
            )
            fill_color_with_alpha = (
                overlay_color[0],
                overlay_color[1],
                overlay_color[2],
                int(255 * overlay_alpha),
            )
            if overlay_type == "mask_pixels":
                if len(pixel_coords) >= 3:
                    polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
                    draw_composite.polygon(
                        polygon_points, outline=fill_color_with_alpha, width=line_width
                    )
            elif overlay_type == "points":
                point_radius = 2
                for x, y in pixel_coords:
                    draw_composite.ellipse(
                        [
                            x - point_radius,
                            y - point_radius,
                            x + point_radius,
                            y + point_radius,
                        ],
                        fill=fill_color_with_alpha,
                    )
        final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert(
            "RGB"
        )
        final_combined_img.save(save_path)
    except FileNotFoundError:
        print(
            f"Error: ECT image file not found at {ect_image_path}. Skipping combined visualization."
        )
    except Exception as e:
        print(f"Error creating combined visualization for {ect_image_path.stem}: {e}")


def rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates 2D coordinates (Nx2 array) around the origin (0,0).
    """
    if coords.size == 0:
        return np.array([])
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    rotated_coords = coords @ rot_matrix.T
    return rotated_coords


##############################
### CORE LOGIC FUNCTIONS ###
##############################


def load_pca_model_data(pca_params_file: Path, processed_data_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels.
    """
    pca_data = {}
    with h5py.File(pca_params_file, "r") as f:
        pca_data["components"] = f["components"][:]
        pca_data["mean"] = f["mean"][:]
        pca_data["explained_variance"] = f["explained_variance"][:]
        pca_data["n_components"] = f.attrs["n_components"]
    with h5py.File(processed_data_file, "r") as f:
        # Load data from the correct H5 keys
        pca_data["original_pca_scores"] = f["pca_scores"][:]
        pca_data["original_flattened_coords"] = f["aligned_coords"][:].reshape(
            f["aligned_coords"].shape[0], -1
        )
        metadata_df = pd.DataFrame(
            {
                col: np.array([s.decode("utf-8") for s in f[f"metadata/{col}"][:]])
                for col in f["metadata"]
            }
        )
        pca_data["original_class_labels"] = metadata_df["official_name"].values

    print(f"Loaded PCA model parameters from {pca_params_file}.")
    print(f"Loaded original PCA scores and labels from {processed_data_file}.")
    return pca_data


def generate_synthetic_pca_samples(pca_data: dict, min_samples_for_augmentation: int):
    """
    Generates synthetic PCA samples for under-represented classes using a SMOTE-like approach.
    """
    print("\nStarting synthetic data generation (SMOTE-like)...")
    original_pca_scores = pca_data["original_pca_scores"]
    original_class_labels = pd.Series(pca_data["original_class_labels"])

    synthetic_X_pca = []
    synthetic_y = []

    class_counts = original_class_labels.value_counts()

    # Determine the target number of samples for each augmented class
    max_real_samples = class_counts.max()
    print(f"Maximum number of real leaves for any class is: {max_real_samples}")

    classes_to_augment = class_counts[class_counts < max_real_samples].index.tolist()
    total_generated_samples = 0

    for class_name in classes_to_augment:
        num_existing_samples = class_counts[class_name]

        # Only augment if the class has enough samples for k-NN
        if num_existing_samples < min_samples_for_augmentation:
            print(
                f"Warning: Class '{class_name}' has too few samples ({num_existing_samples}) for augmentation. Skipping."
            )
            continue

        class_pca_samples = original_pca_scores[original_class_labels == class_name]
        samples_to_generate = max_real_samples - num_existing_samples

        print(
            f"Augmenting class '{class_name}' with {samples_to_generate} samples to reach the target of {max_real_samples}."
        )

        n_neighbors_for_class = min(len(class_pca_samples) - 1, K_NEIGHBORS_SMOTE)
        nn = NearestNeighbors(n_neighbors=n_neighbors_for_class + 1).fit(
            class_pca_samples
        )

        for _ in range(samples_to_generate):
            idx_in_class_samples = np.random.randint(0, len(class_pca_samples))
            sample = class_pca_samples[idx_in_class_samples]
            distances, indices = nn.kneighbors(sample.reshape(1, -1))

            available_neighbors_indices_in_class_pca = indices[0][1:]
            if len(available_neighbors_indices_in_class_pca) == 0:
                continue

            neighbor_idx_in_class_pca_samples = np.random.choice(
                available_neighbors_indices_in_class_pca
            )
            neighbor = class_pca_samples[neighbor_idx_in_class_pca_samples]

            alpha = np.random.rand()
            synthetic_pca_sample = sample + alpha * (neighbor - sample)

            synthetic_X_pca.append(synthetic_pca_sample)
            synthetic_y.append(class_name)
            total_generated_samples += 1

    print(f"Finished generating {total_generated_samples} synthetic samples.")
    return np.array(synthetic_X_pca), synthetic_y


def inverse_transform_pca(
    pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray
):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    """
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data


def process_leaf_for_cnn_output(
    sample_id: str,
    class_label: str,
    flat_coords: np.ndarray,
    ect_calculator: ECT,
    output_dirs: dict,
    metadata_records: list,
    is_real_sample: bool = False,
    apply_random_rotation: bool = False,
    global_ect_min: float = None,
    global_ect_max: float = None,
):
    """
    Processes a single leaf's flattened coordinates to produce masks, ECTs, and combined viz.
    """
    current_metadata = {
        "synthetic_id": sample_id,
        "class_label": class_label,
        "is_processed_valid": False,
        "reason_skipped": "",
        "num_contour_coords": 0,
        "file_shape_mask": "",
        "file_shape_ect": "",
        "file_combined_viz": "",
        "is_real": is_real_sample,
    }

    temp_ect_combined_viz_path = (
        output_dirs["combined_viz"] / f"{sample_id}_ect_temp.png"
    )

    try:
        raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
        current_metadata["num_contour_coords"] = len(raw_contour_coords)

        if apply_random_rotation:
            random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
            processed_contour_coords = rotate_coords_2d(
                raw_contour_coords, random_angle_deg
            )
        else:
            processed_contour_coords = raw_contour_coords.copy()

        if len(np.unique(processed_contour_coords, axis=0)) < 3:
            raise ValueError(
                f"Leaf '{sample_id}' has too few distinct contour points ({len(np.unique(processed_contour_coords, axis=0))}) for ECT calculation."
            )

        G_contour = EmbeddedGraph()
        G_contour.add_cycle(processed_contour_coords)
        original_G_contour_coord_matrix = G_contour.coord_matrix.copy()

        G_contour.center_coordinates(center_type="origin")
        G_contour.transform_coordinates()
        G_contour.scale_coordinates(BOUND_RADIUS)

        if G_contour.coord_matrix.shape[0] < 3 or np.all(G_contour.coord_matrix == 0):
            raise ValueError(
                f"Degenerate contour shape for '{sample_id}' after ECT transformation."
            )

        ect_affine_matrix = find_robust_affine_transformation_matrix(
            original_G_contour_coord_matrix, G_contour.coord_matrix
        )
        transformed_contour_for_mask = apply_transformation_with_affine_matrix(
            processed_contour_coords, ect_affine_matrix
        )
        ect_result = ect_calculator.calculate(G_contour)

        shape_mask_path = output_dirs["shape_masks"] / f"{sample_id}_mask.png"
        shape_ect_path = output_dirs["shape_ects"] / f"{sample_id}_ect.png"
        combined_viz_path = output_dirs["combined_viz"] / f"{sample_id}_combined.png"

        save_grayscale_shape_mask(transformed_contour_for_mask, shape_mask_path)
        save_radial_ect_image(
            ect_result,
            shape_ect_path,
            cmap_name="gray",
            vmin=global_ect_min,
            vmax=global_ect_max,
        )

        save_radial_ect_image(
            ect_result,
            temp_ect_combined_viz_path,
            cmap_name="gray_r",
            vmin=global_ect_min,
            vmax=global_ect_max,
        )

        create_combined_viz_from_images(
            temp_ect_combined_viz_path,
            transformed_contour_for_mask,
            combined_viz_path,
            overlay_color=(0, 0, 0),
            overlay_alpha=1.0,
            overlay_type="mask_pixels",
            line_width=OUTLINE_LINE_WIDTH,
        )

        current_metadata["is_processed_valid"] = True
        current_metadata["file_shape_mask"] = str(
            shape_mask_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR)
        )
        current_metadata["file_shape_ect"] = str(
            shape_ect_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR)
        )
        current_metadata["file_combined_viz"] = str(
            combined_viz_path.relative_to(SYNTHETIC_DATA_OUTPUT_DIR)
        )

    except Exception as e:
        current_metadata["reason_skipped"] = f"Processing failed: {e}"
        print(f"Skipping leaf '{sample_id}' due to error: {e}")

    finally:
        metadata_records.append(current_metadata)
        if temp_ect_combined_viz_path.exists():
            os.remove(temp_ect_combined_viz_path)


def calculate_global_ect_min_max(
    all_flattened_coords: np.ndarray, ect_calculator: ECT, apply_random_rotation: bool
):
    """
    Calculates the global minimum and maximum ECT values across all (real and synthetic) samples.
    """
    print("\n--- Calculating Global ECT Min/Max for consistent visualization ---")
    global_min_val = float("inf")
    global_max_val = float("-inf")
    num_samples = len(all_flattened_coords)

    for i, flat_coords in enumerate(all_flattened_coords):
        if (i + 1) % 100 == 0 or i == num_samples - 1:
            print(f"  Calculating ECT for sample {i + 1}/{num_samples}...")
        try:
            raw_contour_coords = flat_coords.reshape(TOTAL_CONTOUR_COORDS, 2)
            if apply_random_rotation:
                random_angle_deg = np.random.uniform(*RANDOM_ROTATION_RANGE_DEG)
                processed_contour_coords = rotate_coords_2d(
                    raw_contour_coords, random_angle_deg
                )
            else:
                processed_contour_coords = raw_contour_coords.copy()
            if len(np.unique(processed_contour_coords, axis=0)) < 3:
                continue
            G_contour = EmbeddedGraph()
            G_contour.add_cycle(processed_contour_coords)
            G_contour.center_coordinates(center_type="origin")
            G_contour.transform_coordinates()
            G_contour.scale_coordinates(BOUND_RADIUS)
            if G_contour.coord_matrix.shape[0] < 3 or np.all(
                G_contour.coord_matrix == 0
            ):
                continue
            ect_result = ect_calculator.calculate(G_contour)
            global_min_val = min(global_min_val, ect_result.min())
            global_max_val = max(global_max_val, ect_result.max())
        except Exception as e:
            continue
    if global_min_val == float("inf") or global_max_val == float("-inf"):
        print("  Warning: No valid ECT values found. Setting to default [0, 1].")
        global_min_val = 0.0
        global_max_val = 1.0
    elif global_min_val == global_max_val:
        print(
            f"  Warning: All ECT values are identical ({global_min_val}). Adjusting max to avoid issues."
        )
        global_max_val = global_min_val + 1e-6
    print(
        f"  Global ECT Min: {global_min_val:.4f}, Global ECT Max: {global_max_val:.4f}"
    )
    return global_min_val, global_max_val


def main_synthetic_generation(clear_existing_data: bool = True):
    """
    Main function to orchestrate synthetic leaf data generation AND processing of real data.
    """
    np.random.seed(42)
    print("--- Starting Leaf Data Processing and Augmentation Pipeline ---")

    if clear_existing_data and SYNTHETIC_DATA_OUTPUT_DIR.exists():
        print(f"Clearing existing output directory: {SYNTHETIC_DATA_OUTPUT_DIR}")
        shutil.rmtree(SYNTHETIC_DATA_OUTPUT_DIR)
    SYNTHETIC_DATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_MASK_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_SHAPE_ECT_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_COMBINED_VIZ_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories in {SYNTHETIC_DATA_OUTPUT_DIR}.")

    pca_data = load_pca_model_data(PCA_PARAMS_FILE, PROCESSED_DATA_FILE)
    if (
        "original_flattened_coords" not in pca_data
        or pca_data["original_flattened_coords"] is None
    ):
        print(
            "Cannot process real data as 'aligned_coords' was not found in the HDF5 file. Exiting."
        )
        return

    synthetic_X_pca, synthetic_y_labels = generate_synthetic_pca_samples(
        pca_data, MIN_SAMPLES_FOR_AUGMENTATION
    )
    synthetic_flattened_coords = inverse_transform_pca(
        synthetic_X_pca, pca_data["components"], pca_data["mean"]
    )

    all_flattened_coords = np.vstack(
        [pca_data["original_flattened_coords"], synthetic_flattened_coords]
    )

    ect_calculator = ECT(
        num_dirs=NUM_ECT_DIRECTIONS,
        thresholds=ECT_THRESHOLDS,
        bound_radius=BOUND_RADIUS,
    )
    print("Initialized ECT calculator.")

    global GLOBAL_ECT_MIN, GLOBAL_ECT_MAX
    GLOBAL_ECT_MIN, GLOBAL_ECT_MAX = calculate_global_ect_min_max(
        all_flattened_coords, ect_calculator, APPLY_RANDOM_ROTATION
    )

    metadata_records = []

    print("\n--- Processing Original Real Leaf Samples ---")
    num_real_samples = len(pca_data["original_flattened_coords"])
    for i in range(num_real_samples):
        sample_id = f"real_leaf_{i:05d}"
        class_label = pca_data["original_class_labels"][i]
        flat_coords = pca_data["original_flattened_coords"][i]
        if (i + 1) % 50 == 0 or i == num_real_samples - 1:
            print(
                f"Processing real leaf {i + 1}/{num_real_samples} ({sample_id}, Class: {class_label})"
            )
        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            flat_coords,
            ect_calculator,
            {
                "shape_masks": SYNTHETIC_SHAPE_MASK_DIR,
                "shape_ects": SYNTHETIC_SHAPE_ECT_DIR,
                "combined_viz": SYNTHETIC_COMBINED_VIZ_DIR,
            },
            metadata_records,
            is_real_sample=True,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN,
            global_ect_max=GLOBAL_ECT_MAX,
        )

    total_synthetic_samples = len(synthetic_flattened_coords)
    print(f"\n--- Processing {total_synthetic_samples} Synthetic Leaf Samples ---")
    for i in range(total_synthetic_samples):
        sample_id = f"synthetic_leaf_{i:05d}"
        class_label = synthetic_y_labels[i]
        flat_coords = synthetic_flattened_coords[i]
        if (i + 1) % 50 == 0 or i == total_synthetic_samples - 1:
            print(
                f"Processing synthetic leaf {i + 1}/{total_synthetic_samples} ({sample_id}, Class: {class_label})"
            )
        process_leaf_for_cnn_output(
            sample_id,
            class_label,
            flat_coords,
            ect_calculator,
            {
                "shape_masks": SYNTHETIC_SHAPE_MASK_DIR,
                "shape_ects": SYNTHETIC_SHAPE_ECT_DIR,
                "combined_viz": SYNTHETIC_COMBINED_VIZ_DIR,
            },
            metadata_records,
            is_real_sample=False,
            apply_random_rotation=APPLY_RANDOM_ROTATION,
            global_ect_min=GLOBAL_ECT_MIN,
            global_ect_max=GLOBAL_ECT_MAX,
        )

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(SYNTHETIC_METADATA_FILE, index=False)
    print(
        f"\nSaved combined real and synthetic leaf metadata to {SYNTHETIC_METADATA_FILE}"
    )

    print("\n--- Consolidating data for CNN training ---")
    valid_samples_df = metadata_df[metadata_df["is_processed_valid"]].copy()
    if valid_samples_df.empty:
        print("No valid samples processed to create the final CNN dataset. Exiting.")
        return

    X_images = []
    y_labels_raw = []
    is_real_flags = []
    for idx, row in valid_samples_df.iterrows():
        mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_mask"]
        ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_ect"]
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask_array = np.array(mask_img, dtype=np.float32) / 255.0
            ect_img = Image.open(ect_path).convert("L")
            ect_array = np.array(ect_img, dtype=np.float32) / 255.0
            combined_image = np.stack([mask_array, ect_array], axis=-1)
            X_images.append(combined_image)
            y_labels_raw.append(row["class_label"])
            is_real_flags.append(row["is_real"])
        except FileNotFoundError:
            print(f"Warning: Missing image file for {row['synthetic_id']}. Skipping.")
        except Exception as e:
            print(
                f"Error loading or processing images for {row['synthetic_id']}: {e}. Skipping."
            )

    if not X_images:
        print("No images were successfully loaded. The final dataset will be empty.")
        return

    X_images_np = np.array(X_images)
    y_labels_np = np.array(y_labels_raw)
    is_real_flags_np = np.array(is_real_flags)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels_np)
    class_names = label_encoder.classes_
    X_images_shuffled, y_encoded_shuffled, is_real_flags_shuffled = shuffle(
        X_images_np, y_encoded, is_real_flags_np, random_state=42
    )

    final_data = {
        "X_images": X_images_shuffled,
        "y_labels_encoded": y_encoded_shuffled,
        "class_names": class_names,
        "is_real_flags": is_real_flags_shuffled,
        "image_size": IMAGE_SIZE,
        "num_channels": X_images_shuffled.shape[-1],
    }

    with open(FINAL_PREPARED_DATA_FILE, "wb") as f:
        pickle.dump(final_data, f)
    print(
        f"Successfully prepared and saved CNN training data to {FINAL_PREPARED_DATA_FILE}"
    )
    print(
        f"Dataset shape: X_images={X_images_shuffled.shape}, y_labels={y_encoded_shuffled.shape}"
    )
    print(f"Class names: {class_names}")
    print("\n--- Leaf Data Processing and Augmentation Pipeline Completed ---")


if __name__ == "__main__":
    main_synthetic_generation(clear_existing_data=True)
