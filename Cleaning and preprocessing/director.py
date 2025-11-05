import pandas as pd
import numpy as np
import os
import re

# ===============================
# 🔹 CONFIGURATION
# ===============================
INPUT_CSV = r"C:\Users\mahek\Desktop\DATASETS_2025\film-dataset.csv"
OUTPUT_CSV = r"C:\Users\mahek\Desktop\DATASETS_2025\cleaned_film_dataset.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ===============================
# 🧾 1️⃣ EXTRACT
# ===============================
print("📥 Extracting data from CSV...")

try:
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ CSV loaded successfully! Original shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    exit()

# ===============================
# 🧹 2️⃣ TRANSFORM
# ===============================
print("🔄 Cleaning and transforming data...")

# Keep only necessary columns
columns_to_keep = [
    "unique.id", "imdb.id", "title.mixed",
    "prod.country.1.en", "director.1", "genre", "fest.first"
]
available_cols = [col for col in columns_to_keep if col in df.columns]
df = df[available_cols].copy()

# Rename columns
df.rename(columns={
    "unique.id": "movie_id",
    "imdb.id": "imdb_id",
    "title.mixed": "title",
    "prod.country.1.en": "country",
    "director.1": "director",
    "fest.first": "festival"
}, inplace=True)

# Drop almost empty columns (less than 20% filled)
threshold = len(df) * 0.2
df.dropna(axis=1, thresh=threshold, inplace=True)

# Drop rows missing IMDb IDs
df = df[df["imdb_id"].notna() & (df["imdb_id"].astype(str).str.strip() != "")]
print(f"✅ Removed rows with missing IMDb IDs. Remaining rows: {len(df)}")

# Fill missing values
df.fillna({
    "title": "Unknown Title",
    "country": "Unknown",
    "genre": "Unknown",
    "festival": "Not Specified",
    "director": "Unknown"
}, inplace=True)

# ---------------------------------
# NLP-style text cleaning
# ---------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_director(text):
    if pd.isna(text) or str(text).strip() == "":
        return "unknown"
    parts = re.split(r',|;|&| and ', str(text))
    first = parts[0].strip()
    first = re.sub(r'[^a-zA-Z\s]', '', first)
    first = re.sub(r'\s+', ' ', first)
    return first.title() if first else "Unknown"

# Apply cleaning
df["title"] = df["title"].apply(clean_text)
df["country"] = df["country"].apply(lambda x: str(x).title().strip())
df["genre"] = df["genre"].apply(clean_text)
df["festival"] = df["festival"].apply(clean_text)
df["director"] = df["director"].apply(clean_director)

# Drop duplicates
df.drop_duplicates(subset=["imdb_id"], keep="first", inplace=True)
df.drop_duplicates(subset=["title", "director"], keep="first", inplace=True)

# Reorder columns
final_df = df[["movie_id", "imdb_id", "title", "country", "director", "genre", "festival"]]

print(f"✅ Cleaned dataset shape: {final_df.shape}")

# ===============================
# 💾 3️⃣ LOAD
# ===============================
print("💾 Saving cleaned CSV...")

# Save normally (no encoding forced)
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ File saved successfully at:\n{OUTPUT_CSV}")
