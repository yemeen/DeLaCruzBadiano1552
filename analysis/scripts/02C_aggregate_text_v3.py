import pandas as pd
import os
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords

# --- NLTK Downloads (Run once if you haven't) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK 'stopwords' for English...")
    nltk.download('stopwords')
try:
    stopwords.words('spanish')
except LookupError:
    print("Downloading NLTK 'stopwords' for Spanish...")
    nltk.download('stopwords')

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT) # Go up one level from 'scripts' to project root

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs")
OUTPUTS_PROCESSING_DIR = os.path.join(OUTPUTS_BASE_DIR, "master_sheet_processing")

MASTER_SHEET_FILENAME = "FOR NAHUATL REVIEW - Nahuatl names.csv"
MASTER_SHEET_PATH = os.path.join(DATA_DIR, MASTER_SHEET_FILENAME)

VERIFIED_LINKS_CSV = os.path.join(OUTPUTS_PROCESSING_DIR, "verified_subchapter_links.csv")

ENGLISH_TEXTS_FOLDER = os.path.join(DATA_DIR, "texts")
SPANISH_TEXTS_FOLDER = os.path.join(DATA_DIR, "texts_ES")

AGGREGATED_TEXTS_CSV = os.path.join(OUTPUTS_PROCESSING_DIR, "aggregated_plant_texts.csv")

# Ensure output directory exists
os.makedirs(OUTPUTS_PROCESSING_DIR, exist_ok=True)

# --- Helper Functions ---

def normalize_subchapter_code(sc_code: str) -> str:
    """
    Normalizes a subchapter code (e.g., "8j" -> "08j").
    Handles cases like "chapter01" by returning as is.
    """
    match = re.match(r'(\d{1,2})([a-z])', sc_code)
    if match:
        return f"{int(match.group(1)):02d}{match.group(2)}"
    return sc_code

def load_master_sheet(file_path: str) -> pd.DataFrame:
    """
    Loads the master sheet CSV and performs initial data cleaning.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Master sheet file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    cols_to_drop = ['PROBLEMS', 'NOTES', 'REFERENCES']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    string_cols = ['ID', 'official_name', 'type', 'text_subchapters', 'illustrations', 'synonyms', 'label'] 
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('').str.strip()
            
    return df

def parse_semicolon_list(text: any) -> list[str]:
    """
    Parses a semicolon-separated string into a list of cleaned, non-empty strings.
    Handles non-string inputs (like NaN) by returning an empty list.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    return [item.strip() for item in text.split(';') if item.strip()]

# --- Crucial: This function fixes the encoding ---
def fix_double_encoded_accents(text: str) -> str:
    """
    Replaces common double-encoded UTF-8 character sequences (e.g., 'ÃƒÂ¡')
    and also single-misinterpreted Latin-1 characters (e.g., 'Ã¡')
    back to their correct single accented characters (e.g., 'á').
    """
    replacements = {
        # Double-encoded patterns (e.g., UTF-8 -> Latin-1 -> UTF-8)
        'ÃƒÂ¡': 'á', 'ÃƒÂ©': 'é', 'ÃƒÂ­': 'í', 'ÃƒÂ³': 'ó', 'ÃƒÂº': 'ú', 'ÃƒÂ±': 'ñ',
        'ÃƒÂ': 'Á', 'Ãƒâ€°': 'É', 'ÃƒÂ': 'Í', 'Ãƒâ€œ': 'Ó', 'ÃƒÅº': 'Ú', 'Ãƒâ€˜': 'Ñ',
        
        # Single-misinterpreted patterns (e.g., UTF-8 -> Latin-1)
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú', 'Ã±': 'ñ',
        'Ã\xad': 'í', 'Ã\xb1': 'ñ', 'Ã\xa1': 'á', 'Ã\xa9': 'é', 'Ã\xb3': 'ó', 'Ã\xba': 'ú',
        'Ã\x81': 'Á', 'Ã\x89': 'É', 'Ã\x8D': 'Í', 'Ã\x93': 'Ó', 'Ã\x9A': 'Ú', 'Ã\x91': 'Ñ',

        # Inverted punctuation that often gets corrupted
        'Â¿': '¿', 'Â¡': '¡',
        
        # Other less common but possible encodings
        'ÃƒÂ¤': 'ä', 'ÃƒÂ¶': 'ö', 'ÃƒÂ¼': 'ü', 'ÃƒÂ ': 'à', 'ÃƒÂ¨': 'è', 'ÃƒÂ¬': 'ì', 'ÃƒÂ²': 'ò',
        'ÃƒÂ¢': 'â', 'ÃƒÂª': 'ê', 'ÃƒÂ®': 'î', 'ÃƒÂ´': 'ô', 'ÃƒÂ»': 'û', 'ÃƒÂ§': 'ç', 
        'ÃƒÂ¿': 'ÿ', 'ÃƒÂ¾': 'þ', 'ÃƒÅ¸': 'Þ', 'ÃƒÂ¦': 'æ', 'ÃƒËœ': 'Ø', 'Ãƒâ€¦': 'Å', 'ÃƒÂ°': 'ð'
    }
    
    sorted_replacements = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    processed_text = text
    for old, new in sorted_replacements:
        processed_text = processed_text.replace(old, new)
    return processed_text

# --- End fix_double_encoded_accents ---

def get_subchapter_text_content(subchapter_code: str, lang_type: str) -> str | None:
    """
    Finds and reads the content of a subchapter text file for a given language.
    Returns content in lowercase or None if not found/readable.
    Assumes Spanish files are UTF-8 but may contain corrupted characters that are fixed post-read.
    """
    if lang_type == 'english':
        folder_path = ENGLISH_TEXTS_FOLDER
        file_encoding = "utf-8" # Standard for English
    elif lang_type == 'spanish':
        folder_path = SPANISH_TEXTS_FOLDER
        file_encoding = "utf-8" 
    else:
        return None
        
    if not os.path.exists(folder_path):
        print(f"Warning: Text folder for '{lang_type}' not found at '{folder_path}'.")
        return None

    if subchapter_code.startswith("chapter"):
        search_prefix = f"{subchapter_code}_p"
    else:
        search_prefix = f"{normalize_subchapter_code(subchapter_code)}_p" 
        
    matches = [f for f in os.listdir(folder_path) 
               if f.startswith(search_prefix) and f.endswith(".txt")]
    
    if not matches:
        return None
    
    txt_path = os.path.join(folder_path, matches[0])
    if not os.path.exists(txt_path):
        return None
        
    try:
        with open(txt_path, "r", encoding=file_encoding) as f:
            content = f.read()
            if lang_type == 'spanish':
                content = fix_double_encoded_accents(content)
            return content.lower()
    except Exception as e:
        print(f"Error reading file {txt_path} with encoding '{file_encoding}': {e}")
        return None

# --- Text Cleaning and Normalization ---

# --- CRUCIAL CHANGE #1: Build an enhanced synonym-to-official_name map ---
# We now include variations with hyphens replaced by spaces to handle multi-word terms.
GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP = {}
master_df_for_map = load_master_sheet(MASTER_SHEET_PATH) 
for _, row in master_df_for_map.iterrows():
    official_name_lower = row['official_name'].lower()
    
    synonyms_for_row = parse_semicolon_list(str(row['synonyms']))
    morphemes_for_row = parse_semicolon_list(str(row['label']))
    
    all_terms_for_this_row = [official_name_lower] + synonyms_for_row + [unit.split('/')[0].strip().lower() for unit in morphemes_for_row if '/' in unit]

    for term_lower in all_terms_for_this_row:
        if term_lower:
            # Add the original term
            if term_lower not in GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP:
                GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP[term_lower] = official_name_lower
            
            # If the term contains a hyphen, also add a space-separated version.
            # This handles cases where the raw text uses hyphens for a term that
            # is a single, non-hyphenated word in the master sheet (e.g., tlacacamotli)
            if '-' in term_lower:
                space_separated_term = term_lower.replace('-', ' ')
                if space_separated_term not in GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP:
                    GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP[space_separated_term] = official_name_lower
# --- END CRUCIAL CHANGE #1 ---

ENGLISH_STOPWORDS = set(stopwords.words('english'))
SPANISH_STOPWORDS = set(stopwords.words('spanish'))

# --- CRUCIAL CHANGE #2: The new and improved text cleaning function ---
def clean_and_normalize_text(text: str, language: str) -> str:
    """
    Applies text cleaning and normalization steps in a single, robust pass.
    1. Replaces hyphens with spaces to prevent improper word merging.
    2. Replaces known synonyms/terms with their official names.
    3. Removes remaining punctuation and numbers.
    4. Removes stopwords.
    """
    # 1. CRUCIAL FIX: Replace hyphens and other word-separating punctuation with spaces.
    # This prevents 'overheat-xochitl' from becoming 'overheatxochitl' and
    # allows multi-part terms to be correctly recognized.
    processed_text = re.sub(r'[-—]', ' ', text)
    
    # 2. Perform all synonym/term replacements in one pass.
    # Function to get the official name for a matched term
    def replacer(match):
        term = match.group(0).lower()
        return GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP.get(term, term)

    # Sort the terms by length, in descending order.
    # This is critical to ensure that longer, more specific names (e.g., 'tolohoa-xihuitl')
    # are matched before shorter, more general ones (e.g., 'tolohoa').
    # We now use the keys of our enhanced map.
    sorted_terms = sorted(GLOBAL_SYNONYM_TO_OFFICIAL_NAME_MAP.keys(), key=len, reverse=True)
    
    # Create a single, powerful regex pattern from the sorted terms.
    pattern = '|'.join(re.escape(term) for term in sorted_terms)
    
    # Use re.sub with our custom replacer function. We use word boundaries \b to ensure
    # we only match whole words, and the (?i) flag for case-insensitivity.
    processed_text = re.sub(r'\b(' + pattern + r')\b', replacer, processed_text, flags=re.IGNORECASE)

    # 3. Remove any remaining punctuation and numbers
    # We use a more specific regex that doesn't strip the spaces we just created.
    processed_text = re.sub(r'[^\w\s]', '', processed_text)
    processed_text = re.sub(r'\d+', '', processed_text)
    
    # 4. Remove stopwords
    words = processed_text.split()
    if language == 'english':
        words = [word for word in words if word not in ENGLISH_STOPWORDS]
    elif language == 'spanish':
        words = [word for word in words if word not in SPANISH_STOPWORDS]
    
    # 5. Normalize whitespace and join
    processed_text = ' '.join(words).strip()
    
    return processed_text
# --- END CRUCIAL CHANGE #2 ---


# --- Main Aggregation Logic ---
def aggregate_plant_texts():
    print(f"Loading verified links from: {VERIFIED_LINKS_CSV}")
    verified_df = pd.read_csv(VERIFIED_LINKS_CSV)
    
    if 'synonym_found' in verified_df.columns:
        verified_df['synonym_found'] = verified_df['synonym_found'].astype(str).fillna('').str.lower()

    print(f"Loaded {len(verified_df)} verified links.")

    grouped = verified_df.groupby(['ID', 'official_name'])

    aggregated_records = []

    for (plant_id, official_name), group_df in grouped:
        aggregated_english_text = []
        aggregated_spanish_text = []
        
        all_nahuatl_terms = set()
        
        plant_type = group_df['type'].iloc[0]
        illustrations = group_df['illustrations'].iloc[0]
        raw_morphemes_string = str(group_df['morphemes'].iloc[0])

        all_nahuatl_terms.add(official_name.lower())

        parsed_morphemes = parse_semicolon_list(raw_morphemes_string)
        for unit in parsed_morphemes:
            nahuatl_part = unit.split('/')[0].strip().lower()
            if nahuatl_part:
                all_nahuatl_terms.add(nahuatl_part)

        for _, row in group_df.iterrows():
            subchapter_code = row['subchapter_code']
            language = row['language']
            synonym_found = row['synonym_found']

            if synonym_found: 
                all_nahuatl_terms.add(synonym_found)

            raw_subchapter_content = get_subchapter_text_content(subchapter_code, language)
            
            if raw_subchapter_content:
                cleaned_content = clean_and_normalize_text(raw_subchapter_content, language)
                if language == 'english':
                    aggregated_english_text.append(cleaned_content)
                elif language == 'spanish':
                    aggregated_spanish_text.append(cleaned_content)

        aggregated_records.append({
            "ID": plant_id,
            "official_name": official_name,
            "type": plant_type,
            "illustrations": illustrations,
            "morphemes_raw_string": raw_morphemes_string,
            "aggregated_english_text": " ".join(aggregated_english_text).strip(),
            "aggregated_spanish_text": " ".join(aggregated_spanish_text).strip(),
            "all_nahuatl_terms": " ".join(sorted(list(all_nahuatl_terms)))
        })

    aggregated_df = pd.DataFrame(aggregated_records)
    return aggregated_df

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting text aggregation for Plant IDs...")
    
    aggregated_plant_texts_df = aggregate_plant_texts()

    print(f"Generated {len(aggregated_plant_texts_df)} aggregated plant text records.")
    aggregated_plant_texts_df.to_csv(AGGREGATED_TEXTS_CSV, index=False)
    print(f"Saved aggregated plant texts to: {AGGREGATED_TEXTS_CSV}")
    print("Text aggregation complete.")