import pandas as pd
import json
import re
from tqdm import tqdm

# --- CONFIG ---
input_file = r"C:\Users\mahek\Desktop\DATASETS_2025\movies.json"
output_file = r"C:\Users\mahek\Desktop\DATASETS_2025\movies_cleaned.csv"
use_textblob = False   # Set True only if you need spelling correction

if use_textblob:
    from textblob import TextBlob

# --- Step 1️⃣ Extract: Load JSONL file efficiently ---
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading JSON lines"):
        try:
            record = json.loads(line.strip())
            data.append(record)
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(data)
print("✅ JSON Loaded Successfully!")
print("📊 Shape:", df.shape)

# --- Step 2️⃣ Transform: Keep and clean columns ---
keep_cols = ["movie_id", "title", "genres", "actor_ids", "runtime", "year"]
df = df[keep_cols].copy()

def clean_text(text):
    """Clean unwanted characters, commas, and non-ASCII symbols."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)       # remove newlines/tabs
    text = re.sub(r"\s+", " ", text)             # collapse multiple spaces
    text = re.sub(r"[\"\'\[\]\{\}]", "", text)   # remove quotes/brackets
    text = re.sub(r"[^\x00-\x7F]+", "", text)    # remove all non-ASCII chars
    return text.strip()

# Apply cleaning
for col in ["movie_id", "title", "runtime", "year"]:
    df[col] = df[col].apply(clean_text)

# --- Step 3️⃣ Flatten list columns and explode actor_id ---
def to_list(val):
    """Convert lists or bracketed strings into clean Python lists."""
    if isinstance(val, list):
        return [str(v).strip() for v in val]
    elif isinstance(val, str):
        val = re.sub(r"[\[\]']", "", val)
        parts = re.split(r"[,;]", val)
        return [p.strip() for p in parts if p.strip()]
    else:
        return []

# Convert genres to joined string (space-separated)
df["genre"] = df["genres"].apply(lambda x: " ".join(to_list(x)))

# Convert actor_ids to list for explosion
df["actor_id"] = df["actor_ids"].apply(to_list)

# Drop original columns
df.drop(columns=["genres", "actor_ids"], inplace=True)

# 🧨 Explode actor_id → one row per actor
df = df.explode("actor_id", ignore_index=True)

# --- Step 4️⃣ (Optional) Title cleanup ---
def polish_text(text):
    text = clean_text(text)
    text = text.title().strip()
    if use_textblob:
        try:
            text = str(TextBlob(text).correct())
        except Exception:
            pass
    return text

tqdm.pandas(desc="Cleaning titles")
df["title"] = df["title"].progress_apply(polish_text)

# --- Step 5️⃣ Data type normalization ---
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["runtime"] = df["runtime"].str.extract(r"(\d+)").astype(float)

# --- Step 6️⃣ Final cleanup ---
df.drop_duplicates(subset=["movie_id", "actor_id"], inplace=True)
df.dropna(subset=["movie_id", "title", "actor_id"], inplace=True)

# --- Step 7️⃣ Reorder columns ---
df = df[["movie_id", "title", "genre", "actor_id", "runtime", "year"]]

# --- Step 8️⃣ Save output ---
df.to_csv(output_file, index=False, encoding="utf-8")

print("\n✅ ETL Completed Successfully!")
print("📁 Output Saved To:", output_file)
print("📊 Final Shape:", df.shape)
print("📄 Columns:", list(df.columns))