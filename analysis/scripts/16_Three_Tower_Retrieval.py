import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import pickle
import cv2
import os
from pathlib import Path
import json
import random
from collections import defaultdict

# --- Configuration and File Paths ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT_DIR / "outputs"
DATA_DIR = PROJECT_ROOT_DIR / "data"

# Image and Metadata Paths
SYNTHETIC_DATA_OUTPUT_DIR = OUTPUTS_DIR / "synthetic_leaf_data"
SYNTHETIC_METADATA_PATH = SYNTHETIC_DATA_OUTPUT_DIR / "synthetic_metadata.csv"
NAHUATL_NAMES_PATH = DATA_DIR / "FOR NAHUATL REVIEW - Nahuatl names.csv"

# Text Data Paths
ENGLISH_FASTTEXT_PKL = (
    OUTPUTS_DIR
    / "synthetic_text_data"
    / "english"
    / "english_fasttext_synthetic_dataset.pkl"
)
AGGREGATED_TEXTS_CSV = (
    OUTPUTS_DIR / "master_sheet_processing" / "aggregated_plant_texts.csv"
)

# Path to the generated node embeddings
GRAPH_EMBEDDINGS_PKL = (
    OUTPUTS_DIR / "three_tower_embeddings" / "nahuatl_node_embeddings.pkl"
)

# Model and training parameters
IMG_WIDTH, IMG_HEIGHT = 256, 256
EMBEDDING_DIM = 300
EMBEDDING_SIZE = 128
GRAPH_EMBEDDING_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
N_SPLITS = 3
SEED = 42

# Output directory for this experiment
EXPERIMENT_NAME = "three_tower_synthetic_kfold_retrieval"
EXPERIMENT_OUTPUT_DIR = OUTPUTS_DIR / EXPERIMENT_NAME
os.makedirs(EXPERIMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUT_DIR / "models", exist_ok=True)
os.makedirs(EXPERIMENT_OUTPUT_DIR / "metrics", exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# --- Set seed for reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


set_seed(SEED)

# --- Step 1: Loading and Aligning Data ---
print("--- Step 1: Loading and Aligning Data ---")

# Load image metadata
synthetic_df = pd.read_csv(SYNTHETIC_METADATA_PATH)
real_image_metadata = synthetic_df[synthetic_df["is_real"] == True]
real_image_metadata = real_image_metadata[
    real_image_metadata["is_processed_valid"] == True
]

# Load text data and graph embeddings
with open(ENGLISH_FASTTEXT_PKL, "rb") as f:
    text_data_dict = pickle.load(f)
raw_texts_df = pd.read_csv(AGGREGATED_TEXTS_CSV)
with open(GRAPH_EMBEDDINGS_PKL, "rb") as f:
    graph_embeddings_dict = pickle.load(f)

# Align real image, text, and graph data by 'official_name'
aligned_real_data = pd.merge(
    real_image_metadata,
    raw_texts_df,
    left_on="class_label",
    right_on="official_name",
    how="inner",
    suffixes=("_img", "_txt"),
)

# Filter out classes that don't have a graph embedding
aligned_real_data["official_name_clean"] = (
    aligned_real_data["official_name"].str.strip().str.lower()
)
aligned_real_data = aligned_real_data[
    aligned_real_data["official_name_clean"].isin(graph_embeddings_dict.keys())
].reset_index(drop=True)

# Filter out classes with fewer samples than n_splits for valid stratification
counts = aligned_real_data["official_name"].value_counts()
valid_classes = counts[counts >= N_SPLITS].index
aligned_real_data = aligned_real_data[
    aligned_real_data["official_name"].isin(valid_classes)
].reset_index(drop=True)

print(
    f"Number of aligned REAL samples for K-Fold after filtering: {len(aligned_real_data)}"
)

# --- Re-populate data arrays from the filtered dataframe ---
X_real_images = []
X_real_text = []
X_real_graph = []
y_real_labels_plant_ids = []
plant_id_to_label = {
    plant_id: i
    for i, plant_id in enumerate(aligned_real_data["official_name"].unique())
}

for idx, row in aligned_real_data.iterrows():
    ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_ect"]
    mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_mask"]

    ect_img = cv2.imread(str(ect_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if ect_img is not None and mask_img is not None:
        ect_img = cv2.resize(ect_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        stacked_img = np.stack([ect_img, mask_img], axis=0)
        X_real_images.append(stacked_img)

        text_index = raw_texts_df[
            raw_texts_df["official_name"] == row["official_name"]
        ].index[0]
        X_real_text.append(text_data_dict["X_vectors"][text_index])

        graph_embedding = graph_embeddings_dict[row["official_name_clean"]]
        X_real_graph.append(graph_embedding)

        y_real_labels_plant_ids.append(plant_id_to_label[row["official_name"]])

X_real_images = np.array(X_real_images, dtype=np.float32)
X_real_text = np.array(X_real_text, dtype=np.float32)
X_real_graph = np.array(X_real_graph, dtype=np.float32)
y_real_labels_plant_ids = np.array(y_real_labels_plant_ids, dtype=np.int64)

# --------------------- FIXED SYNTHETIC TEXT/TRIPLE CREATION ---------------------
# Build normalized mappings: name -> text vec, name -> graph vec, name -> label
raw_texts_df["official_name_clean"] = (
    raw_texts_df["official_name"].str.strip().str.lower()
)
name2text = dict(zip(raw_texts_df["official_name_clean"], text_data_dict["X_vectors"]))
graph_embeddings_lower = {
    k.strip().lower(): v for k, v in graph_embeddings_dict.items()
}
name2label_lower = {
    k.strip().lower(): v
    for k, v in ((name, lbl) for name, lbl in plant_id_to_label.items())
}

# Synthetic metadata (images only)
synthetic_image_metadata = (
    synthetic_df[synthetic_df["is_real"] == False].copy().reset_index(drop=True)
)
num_available_synthetic_images = len(synthetic_image_metadata)
print(f"Number of available synthetic images: {num_available_synthetic_images}")
print(f"Number of real images: {len(X_real_images)}")

# How many synthetic to try to load (ratio relative to real images)
NUM_SYNTHETIC_RATIO = 10
num_synthetic_to_load = int(len(X_real_images) * NUM_SYNTHETIC_RATIO)

# Build a list of eligible synthetic rows (have text+graph+label and files exist)
eligible_indices = []
eligible_keys = []
for i, row in synthetic_image_metadata.iterrows():
    name = row.get("official_name", row.get("class_label", None))
    if not isinstance(name, str):
        continue
    key = name.strip().lower()
    if (
        key not in name2text
        or key not in graph_embeddings_lower
        or key not in name2label_lower
    ):
        continue
    ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_ect"]
    mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_mask"]
    if not (ect_path.exists() and mask_path.exists()):
        continue
    eligible_indices.append(i)
    eligible_keys.append(key)

print(
    f"Eligible synthetic images with matching text+graph+label: {len(eligible_indices)}"
)

# Sample up
num_to_sample = min(num_synthetic_to_load, len(eligible_indices))
rng = np.random.default_rng(SEED)
if num_to_sample > 0:
    chosen_positions = rng.choice(
        len(eligible_indices), size=num_to_sample, replace=False
    )
    chosen_indices = [eligible_indices[j] for j in chosen_positions]
    chosen_keys = [eligible_keys[j] for j in chosen_positions]
else:
    chosen_indices, chosen_keys = [], []

# Construct aligned synthetic arrays by COPYING the real text/graph for each plant
X_synth_img, X_synth_txt, X_synth_graph, y_synth_labels = [], [], [], []
for idx, key in zip(chosen_indices, chosen_keys):
    row = synthetic_image_metadata.iloc[idx]
    ect_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_ect"]
    mask_path = SYNTHETIC_DATA_OUTPUT_DIR / row["file_shape_mask"]

    ect_img = cv2.imread(str(ect_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if ect_img is None or mask_img is None:
        continue

    # Keep ECT smooth, mask crisp
    ect_img = (
        cv2.resize(ect_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        / 255.0
    )
    mask_img = (
        cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        / 255.0
    )
    stacked = np.stack([ect_img, mask_img], axis=0).astype(np.float32)

    X_synth_img.append(stacked)
    X_synth_txt.append(
        np.asarray(name2text[key], dtype=np.float32)
    )
    X_synth_graph.append(
        np.asarray(graph_embeddings_lower[key], dtype=np.float32)
    )
    y_synth_labels.append(
        np.int64(name2label_lower[key])
    )

X_synthetic_images = np.asarray(X_synth_img, dtype=np.float32)
X_synthetic_text = np.asarray(X_synth_txt, dtype=np.float32)
X_synthetic_graph = np.asarray(X_synth_graph, dtype=np.float32)
y_synthetic_labels = np.asarray(y_synth_labels, dtype=np.int64)

print(
    f"Loaded {len(X_synthetic_images)} aligned synthetic triples (image+copied text+graph)."
)


# --- Step 2: Custom PyTorch Dataset and DataLoader (Modified for 3 modalities) ---
class ThreeTowerRetrievalDataset(Dataset):
    def __init__(self, images, texts, graphs, labels):
        self.images = torch.from_numpy(images)
        self.texts = torch.from_numpy(texts)
        self.graphs = torch.from_numpy(graphs)
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.texts[idx], self.graphs[idx], self.labels[idx]


# --- Step 3: Building the Three-Tower Retrieval Model with PyTorch ---
class ThreeTowerRetrievalModel(nn.Module):
    def __init__(self, embedding_size):
        super(ThreeTowerRetrievalModel, self).__init__()
        self.image_tower = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embedding_size),
        )
        self.image_output_size = self._get_image_output_size()
        self.image_projection = nn.Linear(self.image_output_size, embedding_size)

        self.text_tower = nn.Sequential(
            nn.Conv1d(EMBEDDING_DIM, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.text_projection = nn.Linear(128, embedding_size)

        self.graph_projection = nn.Linear(GRAPH_EMBEDDING_SIZE, embedding_size)

    def _get_image_output_size(self):
        dummy_input = torch.randn(1, 2, IMG_WIDTH, IMG_HEIGHT)
        output = self.image_tower(dummy_input)
        return output.size(1)

    def forward_image(self, image_input):
        features = self.image_tower(image_input)
        embeddings = self.image_projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward_text(self, text_input):
        features = self.text_tower(text_input.permute(0, 2, 1))
        embeddings = self.text_projection(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward_graph(self, graph_input):
        embeddings = self.graph_projection(graph_input)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, image_input, text_input, graph_input):
        img_emb = self.forward_image(image_input)
        txt_emb = self.forward_text(text_input)
        graph_emb = self.forward_graph(graph_input)
        return img_emb, txt_emb, graph_emb


# --- Step 4: Multi-Modal Contrastive Loss Function ---
class LabelAwareMultiModalLoss(nn.Module):
    """
    Multi-positive InfoNCE across modalities using labels.
    For each pair (A,B), builds logits = A @ B^T / tau and treats
    any equal-label pairs as positives.
    """

    def __init__(self, tau: float = 0.07, average_directions: bool = True):
        super().__init__()
        self.tau = tau
        self.average_directions = True

    def _pair_loss(
        self, Za: torch.Tensor, Zb: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # Za, Zb: [B, D], already L2-normalized
        # labels: [B], int
        logits = (Za @ Zb.t()) / self.tau
        # mask of positives for each query (multi-positive)
        pos_mask = (labels[:, None] == labels[None, :]).to(
            logits.device
        )
        # avoid all-false rows
        pos_count = pos_mask.sum(dim=1).clamp_min(1)

        # log-sum-exp over all targets (denominator)
        logZ = torch.logsumexp(logits, dim=1)

        # log-sum-exp over positives only
        logits_pos = logits.masked_fill(~pos_mask, float("-inf"))
        log_pos = torch.logsumexp(logits_pos, dim=1)

        # negative log-likelihood of the positive set (normalized by #positives)
        loss = -(log_pos - logZ) / pos_count
        return loss.mean()

    def forward(self, img_emb, txt_emb, graph_emb, labels):
        # average losses over both directions for each pair (optional but common)
        lt_i2t = self._pair_loss(img_emb, txt_emb, labels)
        lt_t2i = self._pair_loss(txt_emb, img_emb, labels)
        lt_i2g = self._pair_loss(img_emb, graph_emb, labels)
        lt_g2i = self._pair_loss(graph_emb, img_emb, labels)
        lt_t2g = self._pair_loss(txt_emb, graph_emb, labels)
        lt_g2t = self._pair_loss(graph_emb, txt_emb, labels)

        if self.average_directions:
            return (lt_i2t + lt_t2i + lt_i2g + lt_g2i + lt_t2g + lt_g2t) / 6.0
        else:
            return (lt_i2t + lt_i2g + lt_t2g) / 3.0


class MultiModalContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(MultiModalContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, img_emb, txt_emb, graph_emb):
        loss_img_txt = self.contrastive_loss_pair(img_emb, txt_emb)
        loss_img_graph = self.contrastive_loss_pair(img_emb, graph_emb)
        loss_txt_graph = self.contrastive_loss_pair(txt_emb, graph_emb)

        return loss_img_txt + loss_img_graph + loss_txt_graph

    def contrastive_loss_pair(self, emb1, emb2):
        positive_similarity = F.cosine_similarity(emb1, emb2)
        emb1_expanded = emb1.unsqueeze(1)
        emb2_expanded = emb2.unsqueeze(0)
        negative_similarity = F.cosine_similarity(emb1_expanded, emb2_expanded, dim=2)
        negative_similarity = negative_similarity.fill_diagonal_(0.0)
        loss_positive = (1 - positive_similarity).pow(2).mean()
        loss_negative = (
            torch.clamp(negative_similarity - self.margin, min=0).pow(2).mean()
        )
        return loss_positive + loss_negative


# --- Step 5: K-Fold Cross-Validation Loop (Modified for 3 modalities) ---
print("\n--- Step 5: Starting K-Fold Cross-Validation with Synthetic Data ---")
if len(X_real_images) == 0:
    print(
        "Error: No real images were loaded after filtering. Cannot perform K-Fold validation."
    )
elif len(np.unique(y_real_labels_plant_ids)) < N_SPLITS:
    print(
        f"Error: The number of unique classes ({len(np.unique(y_real_labels_plant_ids))}) is less than the number of splits ({N_SPLITS}). Cannot perform stratified K-fold. Please adjust N_SPLITS."
    )
else:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_indices, test_indices) in enumerate(
        skf.split(X_real_images, y_real_labels_plant_ids)
    ):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        set_seed(SEED + fold)

        X_train_real_img = X_real_images[train_indices]
        X_train_real_txt = X_real_text[train_indices]
        X_train_real_graph = X_real_graph[train_indices]

        X_test_real_img = X_real_images[test_indices]
        X_test_real_txt = X_real_text[test_indices]
        X_test_real_graph = X_real_graph[test_indices]

        # Combine real and synthetic for training
        X_train_img_combined = np.concatenate(
            [X_train_real_img, X_synthetic_images], axis=0
        )
        X_train_txt_combined = np.concatenate(
            [X_train_real_txt, X_synthetic_text], axis=0
        )
        X_train_graph_combined = np.concatenate(
            [X_train_real_graph, X_synthetic_graph], axis=0
        )

        y_train_combined = np.concatenate(
            [y_real_labels_plant_ids[train_indices], y_synthetic_labels], axis=0
        )
        train_dataset = ThreeTowerRetrievalDataset(
            X_train_img_combined,
            X_train_txt_combined,
            X_train_graph_combined,
            y_train_combined,
        )
        test_dataset = ThreeTowerRetrievalDataset(
            X_test_real_img,
            X_test_real_txt,
            X_test_real_graph,
            y_real_labels_plant_ids[test_indices],
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = ThreeTowerRetrievalModel(EMBEDDING_SIZE).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = LabelAwareMultiModalLoss(tau=0.07)

        print(
            f"Training model for fold {fold + 1} on {len(X_train_img_combined)} samples (Real+Synthetic)..."
        )

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            for images, texts, graphs, labels in train_loader:
                images, texts, graphs = (
                    images.to(device),
                    texts.to(device),
                    graphs.to(device),
                )
                optimizer.zero_grad()
                img_emb, txt_emb, graph_emb = model(images, texts, graphs)
                loss = criterion(img_emb, txt_emb, graph_emb, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, texts, graphs, labels in test_loader:
                    images, texts, graphs = (
                        images.to(device),
                        texts.to(device),
                        graphs.to(device),
                    )
                    img_emb, txt_emb, graph_emb = model(images, texts, graphs)
                    loss = criterion(img_emb, txt_emb, graph_emb, labels)
                    val_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs."
                    )
                    model.load_state_dict(best_model_state)
                    break

        # --- MODIFIED 'evaluate_fold_metrics' function to save embeddings ---

        def _expected_random_recall_k(
            num_targets: int, num_rel_per_query: torch.Tensor, k: int
        ) -> torch.Tensor:
            """
            Expected Recall@k for a random ranking with R relevant items among N targets:
                E[Recall@k] = 1 - C(N - R, k) / C(N, k)
            Computed per query (vectorized over queries) and returned as a tensor of shape [Q].
            """
            Q = num_rel_per_query.numel()
            N = torch.full((Q,), float(num_targets), dtype=torch.float64)

            R = num_rel_per_query.to(torch.float64)
            k_vec = torch.full((Q,), float(k), dtype=torch.float64)

            expected = torch.zeros_like(R, dtype=torch.float64)

            # No relevant items => expected recall is 0
            has_rel = R > 0
            if has_rel.any():
                n = N[has_rel]
                r = R[has_rel]
                kq = k_vec[has_rel]

                # log C(a, b) = lgamma(a+1) - lgamma(b+1) - lgamma(a-b+1)
                def logC(a, b):
                    return (
                        torch.lgamma(a + 1.0)
                        - torch.lgamma(b + 1.0)
                        - torch.lgamma(a - b + 1.0)
                    )

                # If k > n - r, then C(n - r, k) = 0 -> log = -inf -> expected = 1
                feasible = kq <= (n - r)
                term = torch.full_like(kq, float("-inf"))
                if feasible.any():
                    term[feasible] = logC(
                        n[feasible] - r[feasible], kq[feasible]
                    ) - logC(n[feasible], kq[feasible])

                expected_val = 1.0 - torch.exp(term)
                expected[has_rel] = expected_val

            return expected

        def calculate_metrics(
            similarity_matrix: torch.Tensor,
            query_labels: torch.Tensor,
            target_labels: torch.Tensor,
            ks=(1, 5, 10),
        ):
            """
            Label-aware retrieval metrics with multiple relevant targets per query.
            - similarity_matrix: [Q, T], higher is more similar
            - query_labels:      [Q]   (e.g., official_name-coded ints)
            - target_labels:     [T]
            Returns dict with:
            {
                'Recall@K': {
                    'Recall@1':  {'score': ..., 'above_chance_x': ..., 'chance': ...},
                    'Recall@5':  {...},
                    'Recall@10': {...}
                },
                'mAP': float
            }
            """

            # Ensure CPU tensors for safe indexing ops
            sim = similarity_matrix.detach().cpu()
            qlabs = query_labels.detach().cpu()
            tlabs = target_labels.detach().cpu()

            Q, T = sim.shape
            metrics = {"Recall@K": {}}

            # Count relevant items per query (multi-positives)
            # Note: this can vary per query depending on class frequency in targets.
            num_rel_per_q = torch.stack(
                [(tlabs == qlabs[i]).sum() for i in range(Q)]
            ).to(torch.float64)

            # ----- Recall@k (label-aware) -----
            for k in ks:
                hits = 0
                # Compute expected random recall per query for "× above chance"
                expected_random = _expected_random_recall_k(T, num_rel_per_q, k)
                for i in range(Q):
                    topk_idx = torch.topk(sim[i], k).indices
                    if (tlabs[topk_idx] == qlabs[i]).any():
                        hits += 1

                recall_k = hits / Q
                # Average expected random recall across queries (those with R>0 contribute >0)
                expected_k = expected_random.mean().item()
                above_chance = float(recall_k / max(expected_k, 1e-12))

                metrics["Recall@K"][f"Recall@{k}"] = {
                    "score": float(recall_k),
                    "above_chance_x": above_chance,
                    "chance": expected_k,
                }

            # ----- mAP (label-aware) -----
            # Average Precision per query with multiple relevant items:
            # AP = average over ranks where relevant appears of precision@rank
            ap_vals = []
            for i in range(Q):
                order = torch.argsort(sim[i], descending=True)
                relevant = (tlabs[order] == qlabs[i]).to(torch.float32)
                R_i = relevant.sum().item()
                if R_i <= 0:
                    ap_vals.append(0.0)
                    continue
                precision_at_rank = torch.cumsum(relevant, dim=0) / (
                    torch.arange(T, dtype=torch.float32) + 1.0
                )
                ap = float((precision_at_rank * relevant).sum().item() / R_i)
                ap_vals.append(ap)

            mAP = float(np.mean(ap_vals)) if len(ap_vals) > 0 else 0.0

            return {"Recall@K": metrics["Recall@K"], "mAP": mAP}

        def evaluate_fold_metrics(model, test_loader, device, fold_num, output_dir):
            """
            Runs the model on the test_loader to get embeddings, then computes
            label-aware retrieval metrics for all 6 directions:
            image↔text, image↔graph, text↔graph
            Saves embeddings+labels and returns the metrics dict.
            """
            model.eval()
            all_img_embeddings, all_txt_embeddings, all_graph_embeddings = [], [], []
            all_labels = []

            with torch.no_grad():
                for images, texts, graphs, labels in test_loader:
                    images, texts, graphs = (
                        images.to(device),
                        texts.to(device),
                        graphs.to(device),
                    )
                    img_emb, txt_emb, graph_emb = model(images, texts, graphs)

                    all_img_embeddings.append(img_emb.cpu())
                    all_txt_embeddings.append(txt_emb.cpu())
                    all_graph_embeddings.append(graph_emb.cpu())
                    all_labels.extend(labels.tolist())

            all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
            all_txt_embeddings = torch.cat(all_txt_embeddings, dim=0)
            all_graph_embeddings = torch.cat(all_graph_embeddings, dim=0)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)

            # Save for later visualization (unchanged)
            embeddings_data = {
                "image_embeddings": all_img_embeddings.numpy(),
                "text_embeddings": all_txt_embeddings.numpy(),
                "graph_embeddings": all_graph_embeddings.numpy(),
                "labels": np.array(all_labels),
            }
            with open(
                os.path.join(output_dir, f"fold_{fold_num}_embeddings.pkl"), "wb"
            ) as f:
                pickle.dump(embeddings_data, f)

            # Similarity matrices
            sim_img_txt = F.cosine_similarity(
                all_img_embeddings.unsqueeze(1), all_txt_embeddings.unsqueeze(0), dim=2
            )
            sim_img_graph = F.cosine_similarity(
                all_img_embeddings.unsqueeze(1),
                all_graph_embeddings.unsqueeze(0),
                dim=2,
            )
            sim_txt_graph = F.cosine_similarity(
                all_txt_embeddings.unsqueeze(1),
                all_graph_embeddings.unsqueeze(0),
                dim=2,
            )

            # Because your three modality lists are collected in the same order, the label vectors are the same.
            img_labels = labels_tensor
            txt_labels = labels_tensor
            graph_labels = labels_tensor

            # Compute label-aware metrics in both directions
            metrics = {}
            metrics["image_to_text"] = calculate_metrics(
                sim_img_txt, img_labels, txt_labels
            )
            metrics["text_to_image"] = calculate_metrics(
                sim_img_txt.T.contiguous(), txt_labels, img_labels
            )

            metrics["image_to_graph"] = calculate_metrics(
                sim_img_graph, img_labels, graph_labels
            )
            metrics["graph_to_image"] = calculate_metrics(
                sim_img_graph.T.contiguous(), graph_labels, img_labels
            )

            metrics["text_to_graph"] = calculate_metrics(
                sim_txt_graph, txt_labels, graph_labels
            )
            metrics["graph_to_text"] = calculate_metrics(
                sim_txt_graph.T.contiguous(), graph_labels, txt_labels
            )

            return metrics

        # MODIFIED CALL: Pass fold_num and output_dir to the evaluation function
        fold_metrics.append(
            evaluate_fold_metrics(
                model, test_loader, device, fold, EXPERIMENT_OUTPUT_DIR
            )
        )

    print("\n--- Consolidating Results Across All Folds ---")
    all_metrics = {
        "image_to_text": defaultdict(list),
        "text_to_image": defaultdict(list),
        "image_to_graph": defaultdict(list),
        "graph_to_image": defaultdict(list),
        "text_to_graph": defaultdict(list),
        "graph_to_text": defaultdict(list),
    }

    for fold in fold_metrics:
        for metric_key, metric_data in fold.items():
            all_metrics[metric_key]["mAP"].append(metric_data["mAP"])
            for k in [1, 5, 10]:
                all_metrics[metric_key][f"Recall@{k}_score"].append(
                    metric_data["Recall@K"][f"Recall@{k}"]["score"]
                )
                all_metrics[metric_key][f"Recall@{k}_above_chance"].append(
                    metric_data["Recall@K"][f"Recall@{k}"]["above_chance_x"]
                )

    # Calculate final averages and std deviations
    final_avg_metrics = defaultdict(dict)
    num_test_samples = len(X_real_images) // N_SPLITS
    random_chance_at_1 = 1 / num_test_samples

    for metric_key, data in all_metrics.items():
        final_avg_metrics[metric_key]["Random_Chance_at_1"] = random_chance_at_1
        final_avg_metrics[metric_key]["mAP_avg"] = np.mean(data["mAP"])
        final_avg_metrics[metric_key]["mAP_std"] = np.std(data["mAP"])
        for k in [1, 5, 10]:
            final_avg_metrics[metric_key][f"Recall@{k}_score_avg"] = np.mean(
                data[f"Recall@{k}_score"]
            )
            final_avg_metrics[metric_key][f"Recall@{k}_score_std"] = np.std(
                data[f"Recall@{k}_score"]
            )
            final_avg_metrics[metric_key][f"Recall@{k}_above_chance_avg"] = np.mean(
                data[f"Recall@{k}_above_chance"]
            )

    print("\n--- Final K-Fold Retrieval Metrics with Three Towers ---")
    for metric_key, data in final_avg_metrics.items():
        print(f"\n--- {metric_key.replace('_', ' ').title()} Retrieval ---")
        print(f"Random Chance @1: {data['Random_Chance_at_1']:.4f}")
        for k in [1, 5, 10]:
            avg_score = data[f"Recall@{k}_score_avg"]
            std_score = data[f"Recall@{k}_score_std"]
            above_chance = data[f"Recall@{k}_above_chance_avg"]
            print(
                f" Recall@{k}: {avg_score:.4f} +/- {std_score:.4f} ({above_chance:.2f}x above chance)"
            )
        print(
            f"Mean Average Precision (mAP): {data['mAP_avg']:.4f} +/- {data['mAP_std']:.4f}"
        )

    metrics_path = (
        EXPERIMENT_OUTPUT_DIR / "metrics" / "kfold_retrieval_three_tower_metrics.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(final_avg_metrics, f, indent=4)
    print(f"\nFinal K-fold retrieval metrics with three towers saved to {metrics_path}")


print("\nScript completed successfully.")
