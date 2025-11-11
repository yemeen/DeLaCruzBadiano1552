#######################
### LOAD IN MODULES ###
#######################

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import sys
import json
import torch
import torch.nn as tnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from wordcloud import WordCloud

############################
### CONFIGURATION (ALL PARAMETERS UP FRONT) ###
############################
# --- Note: CONFIGURATION HAS BEEN ADAPTED FOR TEXT DATA ---
# --- Project Structure Configuration ---
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_OUTPUT_BASE_DIR = PROJECT_ROOT_DIR / "outputs" / "fasttext_cnn_analysis"

# --- General Model Training Configuration ---
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0005
K_FOLDS = 5
PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.1
MODEL_IDENTIFIER = "FastText_Morpheme_TextCNN"

# --- Data Input Configuration (Now language-specific) ---
FASTTEXT_PREPARED_DATA_FILES = {
    "english": PROJECT_ROOT_DIR
    / "outputs"
    / "synthetic_text_data"
    / "english"
    / "english_fasttext_synthetic_dataset.pkl",
    "spanish": PROJECT_ROOT_DIR
    / "outputs"
    / "synthetic_text_data"
    / "spanish"
    / "spanish_fasttext_synthetic_dataset.pkl",
}

# --- CORRECTED: Use a single master CSV and specify columns ---
MASTER_AGGREGATED_TEXTS_CSV = (
    PROJECT_ROOT_DIR
    / "outputs"
    / "master_sheet_processing"
    / "aggregated_plant_texts.csv"
)
TEXT_COLUMNS_BY_LANGUAGE = {
    "english": "aggregated_english_text",
    "spanish": "aggregated_spanish_text",
}

# --- Morpheme Classification Configuration ---
MORPHEME_CLASSES = ["Xihuitl", "Xochitl", "Quahuitl", "Patli", "Quilitl"]
target_column_used_for_data = "Nahuatl_Morphemes"

# --- Output Directories Setup (Now language-specific) ---
MODEL_SAVE_BASE_DIR = EXPERIMENT_OUTPUT_BASE_DIR / "trained_models"
METRICS_SAVE_BASE_DIR = EXPERIMENT_OUTPUT_BASE_DIR / "metrics"
WORDCLOUD_DATA_SAVE_BASE_DIR = EXPERIMENT_OUTPUT_BASE_DIR / "wordcloud_data"
WORDCLOUD_FIGURE_SAVE_BASE_DIR = EXPERIMENT_OUTPUT_BASE_DIR / "wordcloud_figures"
os.makedirs(MODEL_SAVE_BASE_DIR, exist_ok=True)
os.makedirs(METRICS_SAVE_BASE_DIR, exist_ok=True)
os.makedirs(WORDCLOUD_DATA_SAVE_BASE_DIR, exist_ok=True)
os.makedirs(WORDCLOUD_FIGURE_SAVE_BASE_DIR, exist_ok=True)


# --- Global Results Storage ---
results_storage = {}

# --- Wordcloud Configuration ---
# Your requested changes are implemented here: using a default colormap
# and defining a custom stop words list.
DEFAULT_WORDCLOUD_COLORMAP = "viridis"
CUSTOM_STOPWORDS = {"also", "v", "si"}


##########################
### DEVICE SETUP ###
##########################

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device for training: {device}")
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Random seeds set for reproducibility.")

# ---------------------------------------------------------------------------- #
#                                                                              #
#                             PYTORCH DATASET & MODEL (ADAPTED FOR TEXT)       #
#                                                                              #
# ---------------------------------------------------------------------------- #


class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for FastText embeddings of text data.
    """

    def __init__(self, embeddings, labels, is_real_flags):
        self.embeddings = embeddings
        self.labels = labels
        self.is_real_flags = is_real_flags

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.is_real_flags[idx]


class TextCNN(tnn.Module):
    def __init__(self, num_classes, max_seq_len, embedding_dim):
        super(TextCNN, self).__init__()
        # --- Using 1D convolutions for text data ---
        self.features = tnn.Sequential(
            tnn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
            tnn.BatchNorm1d(128),
            tnn.ReLU(),
            tnn.MaxPool1d(kernel_size=2),
            tnn.Conv1d(128, 256, kernel_size=3, padding=1),
            tnn.BatchNorm1d(256),
            tnn.ReLU(),
            tnn.MaxPool1d(kernel_size=2),
            tnn.Conv1d(256, 512, kernel_size=3, padding=1),
            tnn.BatchNorm1d(512),
            tnn.ReLU(),
            tnn.MaxPool1d(kernel_size=2),
        )
        # --- Calculate flattened size based on 1D convolutions ---
        with torch.no_grad():
            temp_features_model = self.features.to(device)
            dummy_input = torch.zeros(1, embedding_dim, max_seq_len).to(device)
            flattened_size = temp_features_model(dummy_input).view(1, -1).shape[1]
            temp_features_model.to("cpu")

        self.classifier = tnn.Sequential(
            tnn.Flatten(),
            tnn.Linear(flattened_size, 512),
            tnn.ReLU(),
            tnn.Dropout(0.5),
            tnn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def save_model_checkpoint(
    model, optimizer, epoch, accuracy, model_identifier, target_column, fold_idx, lang
):
    filepath = (
        MODEL_SAVE_BASE_DIR
        / lang
        / f"{model_identifier}_fold{fold_idx}_best_{target_column}.pth"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
    }
    torch.save(state, filepath)
    print(
        f"  --> Saved best model for Fold {fold_idx} ({lang}) (Accuracy: {accuracy:.4f}) at {filepath}"
    )
    return filepath


def get_grad_cam_heatmap(model, target_layer, input_tensor, target_category_idx):
    """
    Generates Grad-CAM heatmap for a given input tensor and target category.
    """
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_out, grad_in):
        nonlocal gradients
        gradients = grad_out[0]

    hook_handle_fwd = target_layer.register_forward_hook(forward_hook)
    hook_handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    output = model(input_tensor)

    one_hot_output = torch.zeros_like(output)
    one_hot_output[:, target_category_idx] = 1
    model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)

    hook_handle_fwd.remove()
    hook_handle_bwd.remove()

    if activations is None or gradients is None:
        return None

    activations = activations.detach().cpu().squeeze(0)
    gradients = gradients.detach().cpu().squeeze(0)

    weights = torch.mean(gradients, dim=-1, keepdim=True)
    cam = weights * activations
    heatmap = torch.sum(cam, dim=0)

    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) + 1e-8

    return heatmap.numpy()


# ---------------------------------------------------------------------------- #
#                                                                              #
#                 MAIN PIPELINE FOR EACH LANGUAGE                              #
#                                                                              #
# ---------------------------------------------------------------------------- #

# Load the master CSV once
try:
    master_texts_df = pd.read_csv(MASTER_AGGREGATED_TEXTS_CSV)
    print(f"Loaded master texts CSV from: {MASTER_AGGREGATED_TEXTS_CSV}")
except FileNotFoundError:
    print(
        f"CRITICAL ERROR: Master texts CSV not found at {MASTER_AGGREGATED_TEXTS_CSV}. Exiting."
    )
    sys.exit(1)

for language in ["english", "spanish"]:
    print("\n" + "=" * 80)
    print(f"--- STARTING ANALYSIS FOR {language.upper()} TEXTS ---")
    print("=" * 80)

    # --- Data Input and Preparation ---
    print(f"\n--- Loading {language} data ---")
    try:
        with open(FASTTEXT_PREPARED_DATA_FILES[language], "rb") as f:
            final_data = pickle.load(f)

        EMBEDDING_DIM = final_data["embedding_dim"]
        MAX_SEQUENCE_LENGTH = final_data["max_sequence_length"]
        X_embeddings_padded = final_data["X_vectors"]
        y_labels_encoded_morphemes = final_data["y_labels"]
        is_real_flags = final_data["is_real_flags"]

        print(f"Loaded {language} data shape: {X_embeddings_padded.shape}")

        X_embeddings_tensor = (
            torch.from_numpy(X_embeddings_padded).float().permute(0, 2, 1)
        )
        y_morpheme_tensor = torch.from_numpy(y_labels_encoded_morphemes).float()
        is_real_flag_tensor = torch.from_numpy(is_real_flags).bool()

    except FileNotFoundError:
        print(f"Error: Data file not found for {language}. Skipping this language.")
        continue
    except Exception as e:
        print(
            f"An error occurred while loading the {language} data: {e}. Skipping this language."
        )
        continue

    # --- Load original text for visualization later ---
    try:
        text_column_name = TEXT_COLUMNS_BY_LANGUAGE[language]
        original_texts_subset = master_texts_df[
            master_texts_df[text_column_name].fillna("").str.strip() != ""
        ]
        real_sample_indices = np.where(is_real_flags)[0]
        original_texts = original_texts_subset.iloc[real_sample_indices][
            text_column_name
        ].reset_index(drop=True)
        print(
            f"Loaded {len(original_texts)} original {language} texts for visualization."
        )
    except Exception as e:
        print(
            f"Warning: Could not load original texts for {language} from column '{text_column_name}'. Cannot perform Grad-CAM visualization. Error: {e}"
        )
        original_texts = None

    # --- PyTorch Training and Evaluation ---
    print(
        f"\n--- Performing PyTorch CNN with {K_FOLDS}-Fold Cross-Validation on Morphemes for {language} ---"
    )

    real_original_indices_global = torch.where(is_real_flag_tensor)[0].cpu().numpy()
    X_original_embeddings_for_skf = X_embeddings_tensor[real_original_indices_global]
    y_original_for_skf = y_morpheme_tensor[real_original_indices_global]

    kf_pytorch = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    all_predictions_logits = []
    saved_model_paths_per_fold = [None] * K_FOLDS

    # --- Calculate class weights for imbalanced dataset ---
    all_training_labels_for_weights = y_morpheme_tensor.cpu().numpy()
    class_counts = np.sum(all_training_labels_for_weights, axis=0)
    total_samples = len(all_training_labels_for_weights)
    num_morpheme_classes = len(MORPHEME_CLASSES)
    morpheme_class_weights = total_samples / (num_morpheme_classes * class_counts)
    class_weights_tensor = torch.tensor(morpheme_class_weights, dtype=torch.float).to(
        device
    )

    for fold_idx, (train_original_real_indices, val_original_real_indices) in enumerate(
        kf_pytorch.split(X_original_embeddings_for_skf.cpu().numpy())
    ):
        # print(f"\n  --- Fold {fold_idx + 1}/{K_FOLDS} for {language} ---")

        # Validation data for this fold
        X_val_embed_fold = X_original_embeddings_for_skf[val_original_real_indices]
        y_val_fold = y_original_for_skf[val_original_real_indices]
        val_dataset = TextDataset(
            X_val_embed_fold,
            y_val_fold,
            torch.ones_like(y_val_fold[:, 0], dtype=torch.bool),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Training data for this fold (including synthetic samples)
        synthetic_indices = torch.where(~is_real_flag_tensor)[0].cpu().numpy()
        global_real_train_indices = real_original_indices_global[
            train_original_real_indices
        ]
        all_training_indices_global = np.concatenate(
            (global_real_train_indices, synthetic_indices)
        )

        X_train_embed_fold_tensor = X_embeddings_tensor[all_training_indices_global]
        y_train_fold_tensor = y_morpheme_tensor[all_training_indices_global]
        is_real_train_fold_tensor = is_real_flag_tensor[all_training_indices_global]

        train_dataset = TextDataset(
            X_train_embed_fold_tensor, y_train_fold_tensor, is_real_train_fold_tensor
        )
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        model = TextCNN(
            num_classes=num_morpheme_classes,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            embedding_dim=EMBEDDING_DIM,
        ).to(device)
        criterion = tnn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_overall_accuracy_for_saving_this_fold = 0.0
        saved_model_path_this_fold = None

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            for embeddings, labels, _ in train_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * embeddings.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                val_true = []
                val_pred = []
                for embeddings, labels, _ in val_loader:
                    embeddings, labels = embeddings.to(device), labels.to(device)
                    outputs = model(embeddings)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * embeddings.size(0)
                    predictions = (torch.sigmoid(outputs) > 0.5).long()
                    val_true.append(labels.cpu().numpy())
                    val_pred.append(predictions.cpu().numpy())

                avg_train_loss = running_loss / len(train_loader.dataset)
                avg_val_loss = val_loss / len(val_loader.dataset)
                val_true_arr = np.concatenate(val_true, axis=0)
                val_pred_arr = np.concatenate(val_pred, axis=0)
                val_accuracy = f1_score(
                    val_true_arr, val_pred_arr, average="micro", zero_division=0
                )
                # print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val F1: {val_accuracy:.4f} (Real Samples)")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                if val_accuracy > best_overall_accuracy_for_saving_this_fold:
                    best_overall_accuracy_for_saving_this_fold = val_accuracy
                    saved_model_path_this_fold = save_model_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        best_overall_accuracy_for_saving_this_fold,
                        MODEL_IDENTIFIER,
                        target_column_used_for_data,
                        fold_idx,
                        language,
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve == PATIENCE:
                    print(
                        f"  Early stopping at epoch {epoch + 1} due to no improvement in validation loss."
                    )
                    break

        model.load_state_dict(best_model_wts)
        saved_model_paths_per_fold[fold_idx] = saved_model_path_this_fold

        model.eval()
        fold_predictions_logits = []
        real_dataset_for_pred = TextDataset(
            X_original_embeddings_for_skf,
            y_original_for_skf,
            torch.ones_like(y_original_for_skf[:, 0], dtype=torch.bool),
        )
        real_loader_for_pred = DataLoader(
            real_dataset_for_pred, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for embeddings_batch, _, _ in real_loader_for_pred:
                embeddings_batch = embeddings_batch.to(device)
                outputs = model(embeddings_batch)
                fold_predictions_logits.append(outputs.cpu().numpy())
        all_predictions_logits.append(np.concatenate(fold_predictions_logits, axis=0))

    # --- Final Ensemble Evaluation on ALL REAL Samples ---
    print(f"\n--- Final Ensemble Evaluation on ALL REAL Samples for {language} ---")
    averaged_logits = np.mean(np.array(all_predictions_logits), axis=0)
    final_predictions_binary = (
        (torch.sigmoid(torch.from_numpy(averaged_logits)) > 0.5).long().numpy()
    )
    final_true_labels_binary = y_original_for_skf.cpu().numpy()

    micro_f1 = f1_score(
        final_true_labels_binary,
        final_predictions_binary,
        average="micro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        final_true_labels_binary,
        final_predictions_binary,
        average="macro",
        zero_division=0,
    )
    print(
        f"\n--- Overall Micro F1 Score ({MODEL_IDENTIFIER} Ensemble, {language}): {micro_f1:.4f} ---"
    )
    print(
        f"--- Overall Macro F1 Score ({MODEL_IDENTIFIER} Ensemble, {language}): {macro_f1:.4f} ---"
    )

    print(f"\n--- Classification Report ({MODEL_IDENTIFIER} Ensemble, {language}) ---")
    report_dict = classification_report(
        final_true_labels_binary,
        final_predictions_binary,
        target_names=MORPHEME_CLASSES,
        zero_division=0,
        output_dict=True,
    )
    print(
        classification_report(
            final_true_labels_binary,
            final_predictions_binary,
            target_names=MORPHEME_CLASSES,
            zero_division=0,
        )
    )

    metrics_output_dir = METRICS_SAVE_BASE_DIR / language
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_output_path = (
        metrics_output_dir
        / f"{MODEL_IDENTIFIER}_classification_report_{target_column_used_for_data}.json"
    )
    with open(metrics_output_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Classification report saved to: {metrics_output_path}")

    cm_real_pt_multilabel = multilabel_confusion_matrix(
        final_true_labels_binary, final_predictions_binary
    )
    fig, axes = plt.subplots(
        nrows=1, ncols=len(MORPHEME_CLASSES), figsize=(4 * len(MORPHEME_CLASSES), 4)
    )
    if len(MORPHEME_CLASSES) == 1:
        axes = [axes]
    for i, cm in enumerate(cm_real_pt_multilabel):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["Not Predicted", "Predicted"],
            yticklabels=["Not True", "True"],
        )
        axes[i].set_title(f"CM for {MORPHEME_CLASSES[i]}", fontsize=12)
        axes[i].set_xlabel("Predicted label")
        axes[i].set_ylabel("True label")

    plt.tight_layout()
    plt.savefig(
        metrics_output_dir
        / f"{MODEL_IDENTIFIER}_ConfusionMatrices_{target_column_used_for_data}.png",
        dpi=300,
    )
    plt.close()

    # ---------------------------------------------------------------------------- #
    #                                                                              #
    #               GENERATE GRAD-CAM WORD CLOUD DATA FOR EACH CLASS               #
    #                                                                              #
    # ---------------------------------------------------------------------------- #
    print(f"\n--- Starting Grad-CAM Word Importance Analysis for {language} ---")

    if not saved_model_paths_per_fold[0]:
        print(
            f"WARNING: Skipping word cloud generation for {language} because no model was saved for the first fold."
        )
    if original_texts is None or original_texts.empty:
        print(
            f"WARNING: Skipping word cloud generation for {language} because no original texts could be loaded. The DataFrame for this language is empty or an error occurred."
        )

    if (
        saved_model_paths_per_fold[0]
        and original_texts is not None
        and not original_texts.empty
    ):
        best_model_path = saved_model_paths_per_fold[0]
        print(
            f"INFO: Best model for {language} found at {best_model_path}. Proceeding with Grad-CAM."
        )
        model_grad_cam = TextCNN(
            num_classes=num_morpheme_classes,
            max_seq_len=MAX_SEQUENCE_LENGTH,
            embedding_dim=EMBEDDING_DIM,
        ).to(device)
        state_dict = torch.load(best_model_path, map_location=device)[
            "model_state_dict"
        ]
        model_grad_cam.load_state_dict(state_dict)
        model_grad_cam.eval()
        target_layer = model_grad_cam.features[-3]
        class_word_importances = {cls: {} for cls in MORPHEME_CLASSES}

        for sample_idx in range(len(y_original_for_skf)):
            input_embedding_tensor = (
                X_original_embeddings_for_skf[sample_idx].unsqueeze(0).to(device)
            )
            text_to_visualize = original_texts.iloc[sample_idx]
            words = text_to_visualize.split()
            true_labels = final_true_labels_binary[sample_idx]

            for class_idx, has_label in enumerate(true_labels):
                if has_label:
                    class_name = MORPHEME_CLASSES[class_idx]
                    heatmap = get_grad_cam_heatmap(
                        model_grad_cam, target_layer, input_embedding_tensor, class_idx
                    )

                    if heatmap is not None:
                        heatmap_resized = np.interp(
                            np.linspace(0, 1, len(words)),
                            np.linspace(0, 1, len(heatmap)),
                            heatmap,
                        )
                        for word, score in zip(words, heatmap_resized):
                            cleaned_word = word.lower().strip('.,-!?:;"()')
                            # --- CORRECTED: Filter out custom stop words ---
                            if cleaned_word and cleaned_word not in CUSTOM_STOPWORDS:
                                class_word_importances[class_name][cleaned_word] = (
                                    class_word_importances[class_name].get(
                                        cleaned_word, 0
                                    )
                                    + score
                                )

        wordcloud_data_output_dir = WORDCLOUD_DATA_SAVE_BASE_DIR / language
        os.makedirs(wordcloud_data_output_dir, exist_ok=True)
        wordcloud_data_output_path = (
            wordcloud_data_output_dir / "grad_cam_word_importances.pkl"
        )
        with open(wordcloud_data_output_path, "wb") as f:
            pickle.dump(class_word_importances, f)

        print(
            f"INFO: Aggregated Grad-CAM word importance data for {language} saved to: {wordcloud_data_output_path}"
        )

        # ---------------------------------------------------------------------------- #
        #                                                                              #
        #                     GENERATE AND SAVE WORD CLOUD FIGURES                     #
        #                                                                              #
        # ---------------------------------------------------------------------------- #
        print(f"\n--- Generating and Saving Word Cloud Figures for {language} ---")

        wordcloud_figure_output_dir = WORDCLOUD_FIGURE_SAVE_BASE_DIR / language
        os.makedirs(wordcloud_figure_output_dir, exist_ok=True)

        for class_name, word_scores in class_word_importances.items():
            if word_scores:
                wc = WordCloud(
                    background_color="white",
                    width=800,
                    height=400,
                    max_words=100,
                    collocations=False,
                    colormap=DEFAULT_WORDCLOUD_COLORMAP,
                ).generate_from_frequencies(word_scores)

                plt.figure(figsize=(8, 4), dpi=300)
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")

                figure_filename = (
                    wordcloud_figure_output_dir / f"wordcloud_gradcam_{class_name}.png"
                )
                plt.savefig(figure_filename, bbox_inches="tight")
                plt.close()
                print(
                    f"SUCCESS: Word cloud for {class_name} saved to: {figure_filename}"
                )
            else:
                print(
                    f"WARNING: No word data found for class: {class_name}. Skipping word cloud generation."
                )
    else:
        print(
            f"\nINFO: Word cloud generation for {language} skipped. See warnings above for details."
        )

print("\n--- Full pipeline for all languages completed ---")
