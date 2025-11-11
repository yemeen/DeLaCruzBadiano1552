import pandas as pd
import numpy as np
import os
import fasttext
from tqdm import tqdm
import urllib.request
import gzip

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUTS_PROCESSING_DIR = os.path.join(OUTPUTS_BASE_DIR, "master_sheet_processing")

# --- File Paths ---
AGGREGATED_TEXTS_CSV = os.path.join(
    OUTPUTS_PROCESSING_DIR, "aggregated_plant_texts.csv"
)
# Paths to the two FastText models you need to download
FASTTEXT_EN_MODEL_PATH = os.path.join(DATA_DIR, "cc.en.300.bin")
FASTTEXT_ES_MODEL_PATH = os.path.join(DATA_DIR, "cc.es.300.bin")
# Output files
EMBEDDINGS_EN_OUTPUT_PATH = os.path.join(
    OUTPUTS_PROCESSING_DIR, "english_text_embeddings.npy"
)
EMBEDDINGS_ES_OUTPUT_PATH = os.path.join(
    OUTPUTS_PROCESSING_DIR, "spanish_text_embeddings.npy"
)
TEXT_LABELS_OUTPUT_PATH = os.path.join(
    OUTPUTS_PROCESSING_DIR, "text_labels_for_embeddings.csv"
)

# Ensure output directory exists
os.makedirs(OUTPUTS_PROCESSING_DIR, exist_ok=True)

# --- Hyperparameters for the Text CNN ---
MAX_SEQUENCE_LENGTH = (
    256
)
EMBEDDING_DIM = 300


def download_fasttext_model(url: str, filepath: str) -> None:
    """
    Downloads a FastText model from the given URL to the specified filepath.
    Shows progress bar during download.
    """
    print(f"Downloading {os.path.basename(filepath)}...")
    print(f"From: {url}")

    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 // total_size, 100)
        print(f"\rProgress: {percent}%", end="")

    try:
        urllib.request.urlretrieve(url, filepath, download_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        raise


# --- Helper Function to Convert Text to Vector Sequences ---
def text_to_vectors(text: str, model, max_len: int) -> np.ndarray:
    """
    Converts a string of text into a sequence of word embedding vectors.
    Pads or truncates the sequence to a fixed length.
    """
    # Handles both NaN (float) and empty strings
    if not isinstance(text, str) or not text.strip():
        # Return a zero-padded array for empty or invalid texts
        return np.zeros((max_len, EMBEDDING_DIM), dtype=np.float32)

    tokens = text.split()
    vectors = [model.get_word_vector(token) for token in tokens]

    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    elif len(vectors) < max_len:
        padding_needed = max_len - len(vectors)
        padding = np.zeros((padding_needed, EMBEDDING_DIM), dtype=np.float32)
        vectors = np.vstack((vectors, padding))

    return np.array(vectors, dtype=np.float32)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Loading aggregated texts from: {AGGREGATED_TEXTS_CSV}")
    try:
        aggregated_df = pd.read_csv(AGGREGATED_TEXTS_CSV)
    except FileNotFoundError:
        print(
            f"Error: The file '{AGGREGATED_TEXTS_CSV}' was not found. "
            "Please run the aggregation script first."
        )
        exit()

    print(f"Loaded {len(aggregated_df)} aggregated text records.")

    # --- Step 1: Load the two FastText models (download if needed) ---
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(FASTTEXT_EN_MODEL_PATH):
        print(f"\nEnglish FastText model not found. Downloading...")
        download_fasttext_model(
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz",
            FASTTEXT_EN_MODEL_PATH + ".gz",
        )
        print("Extracting English model...")
        with gzip.open(FASTTEXT_EN_MODEL_PATH + ".gz", "rb") as f_in:
            with open(FASTTEXT_EN_MODEL_PATH, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(FASTTEXT_EN_MODEL_PATH + ".gz")

    if not os.path.exists(FASTTEXT_ES_MODEL_PATH):
        print(f"\nSpanish FastText model not found. Downloading...")
        download_fasttext_model(
            "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.es.300.bin.gz",
            FASTTEXT_ES_MODEL_PATH + ".gz",
        )
        print("Extracting Spanish model...")
        with gzip.open(FASTTEXT_ES_MODEL_PATH + ".gz", "rb") as f_in:
            with open(FASTTEXT_ES_MODEL_PATH, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(FASTTEXT_ES_MODEL_PATH + ".gz")

    print(f"\nLoading English FastText model from: {FASTTEXT_EN_MODEL_PATH}")
    ft_en_model = fasttext.load_model(FASTTEXT_EN_MODEL_PATH)

    print(f"Loading Spanish FastText model from: {FASTTEXT_ES_MODEL_PATH}")
    ft_es_model = fasttext.load_model(FASTTEXT_ES_MODEL_PATH)

    # --- Step 2: Vectorize English and Spanish texts separately ---
    print(f"\nVectorizing English texts and padding to length {MAX_SEQUENCE_LENGTH}...")
    vectorized_english_texts = []
    # Note: No .fillna('') here, the helper function handles it
    for _, row in tqdm(aggregated_df.iterrows(), total=len(aggregated_df)):
        vectors = text_to_vectors(
            row["aggregated_english_text"], ft_en_model, MAX_SEQUENCE_LENGTH
        )
        vectorized_english_texts.append(vectors)
    final_en_embeddings_array = np.stack(vectorized_english_texts)

    print(f"\nVectorizing Spanish texts and padding to length {MAX_SEQUENCE_LENGTH}...")
    vectorized_spanish_texts = []
    for _, row in tqdm(aggregated_df.iterrows(), total=len(aggregated_df)):
        vectors = text_to_vectors(
            row["aggregated_spanish_text"], ft_es_model, MAX_SEQUENCE_LENGTH
        )
        vectorized_spanish_texts.append(vectors)
    final_es_embeddings_array = np.stack(vectorized_spanish_texts)

    # --- Step 3: Save the final output ---
    print("\nVectorization complete.")
    print(f"Final English embeddings array shape: {final_en_embeddings_array.shape}")
    print(f"Final Spanish embeddings array shape: {final_es_embeddings_array.shape}")

    np.save(EMBEDDINGS_EN_OUTPUT_PATH, final_en_embeddings_array)
    print(f"Saved English text embeddings to: {EMBEDDINGS_EN_OUTPUT_PATH}")

    np.save(EMBEDDINGS_ES_OUTPUT_PATH, final_es_embeddings_array)
    print(f"Saved Spanish text embeddings to: {EMBEDDINGS_ES_OUTPUT_PATH}")

    # We only need to save the labels once
    labels_df = aggregated_df[["ID", "illustrations", "morphemes_raw_string"]]
    labels_df.to_csv(TEXT_LABELS_OUTPUT_PATH, index=False)
    print(f"Saved corresponding labels to: {TEXT_LABELS_OUTPUT_PATH}")

    print(
        "\nText vectorization complete. You now have two separate input streams for your two-tower CNN."
    )
