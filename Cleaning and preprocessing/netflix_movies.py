"""
netflix_etl_clean_full.py
------------------------------------
Robust ETL cleaner for Netflix-style datasets.
- Keeps only 'Movies'
- Drops unneeded columns (description, rating, date_added)
- Cleans nulls, duplicates, and invalid data
- Removes rows where 'cast' is 'Unknown'
- Normalizes text and genres
- SSIS / SQL Server safe (no line breaks or commas)
- Prints progress to console (no external log file)
------------------------------------
"""

import pandas as pd
import numpy as np
import os
import sys
import traceback
import re
from datetime import datetime

# ================================================================
# 🔧 CONFIGURATION
# ================================================================
INPUT_PATH = r"C:\Users\mahek\Desktop\DATASETS_2025\netflix_titles.csv"
OUTPUT_PATH = r"C:\Users\mahek\Desktop\DATASETS_2025\netflix_cleaned.csv"

# ================================================================
# 🧩 HELPER: LOGGING
# ================================================================
def log(msg):
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}")

# ================================================================
# 🧼 CLEANING HELPERS
# ================================================================
def clean_text_basic(text):
    """Clean text: remove special chars, trim spaces, normalize commas."""
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'[\r\n]+', ' ', text)         # Remove newlines
    text = re.sub(r'\s+', ' ', text)             # Normalize spaces
    text = text.replace(',', ';')                # Replace commas to avoid CSV split
    text = text.encode('ascii', 'ignore').decode('utf-8')  # Strip non-ASCII
    return text

def clean_genres(text):
    """Normalize genres (listed_in) for DB insertion."""
    if pd.isna(text):
        return "Unknown"
    text = str(text).lower().strip()
    text = text.replace(',', ';')  # Replace commas between genres
    text = re.sub(r'\s*;\s*', '; ', text)
    return text.title()

# ================================================================
# 🚀 MAIN PIPELINE
# ================================================================
def main():
    log("=== NETFLIX ETL CLEAN START ===")

    # ----------------------------
    # 1️⃣ EXTRACT: READ DATASET
    # ----------------------------
    try:
        df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_PATH, encoding="latin1")
    except Exception as e:
        log(f"❌ Error reading file: {e}")
        raise

    log(f"Loaded dataset: {len(df)} rows × {len(df.columns)} columns")

    # ----------------------------
    # 2️⃣ DROP UNUSED COLUMNS
    # ----------------------------
    drop_cols = ["description", "rating", "date_added"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing_drops, inplace=True, errors="ignore")
    log(f"Dropped columns: {', '.join(existing_drops) if existing_drops else 'None'}")

    # ----------------------------
    # 3️⃣ KEEP ONLY MOVIES
    # ----------------------------
    if "type" in df.columns:
        before = len(df)
        df["type"] = df["type"].astype(str).str.strip().str.lower()
        df = df[df["type"] == "movie"].copy()
        log(f"Filtered only 'Movies': kept {len(df)} rows (removed {before - len(df)})")
    else:
        log("⚠️ No 'type' column found — skipping filter")

    # ----------------------------
    # 4️⃣ STANDARDIZE STRINGS
    # ----------------------------
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(["\\N", "N/A", "NULL", "None", "?"], pd.NA, inplace=True)

    # ----------------------------
    # 5️⃣ HANDLE MISSING VALUES
    # ----------------------------
    critical_cols = [c for c in ["show_id", "title"] if c in df.columns]
    before_drop = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    log(f"Dropped {before_drop - len(df)} rows missing {critical_cols}")

    fill_defaults = {
        "director": "Unknown",
        "cast": "Unknown",
        "country": "Unknown",
        "listed_in": "Unknown",
        "duration": "Unknown",
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col].fillna(val, inplace=True)

    # ----------------------------
    # 6️⃣ REMOVE 'Unknown' CAST
    # ----------------------------
    if "cast" in df.columns:
        before = len(df)
        df["cast"] = df["cast"].astype(str)
        df = df[df["cast"].str.lower() != "unknown"].copy()
        log(f"Dropped {before - len(df)} rows with cast='Unknown'")

    # ----------------------------
    # 7️⃣ CLEAN NUMERIC COLUMNS
    # ----------------------------
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
        median_year = int(df["release_year"].dropna().median()) if not df["release_year"].dropna().empty else 2000
        df["release_year"].fillna(median_year, inplace=True)
        df["release_year"] = df["release_year"].astype(int)

    # ----------------------------
    # 8️⃣ TEXT & GENRE NORMALIZATION
    # ----------------------------
    text_cols = [c for c in ["title", "director", "cast", "country", "duration"] if c in df.columns]
    for col in text_cols:
        df[col] = df[col].apply(clean_text_basic)

    if "listed_in" in df.columns:
        df["listed_in"] = df["listed_in"].apply(clean_genres)

    # ----------------------------
    # 9️⃣ REMOVE DUPLICATES
    # ----------------------------
    subset_cols = [c for c in ["title", "director", "release_year"] if c in df.columns]
    before_dupes = len(df)
    df.drop_duplicates(subset=subset_cols, keep="first", inplace=True)
    log(f"Removed {before_dupes - len(df)} duplicate rows")

    # ----------------------------
    # 🔟 SORT & FORMAT FOR DB
    # ----------------------------
    sort_cols = [c for c in ["release_year", "title"] if c in df.columns]
    if sort_cols:
        df.sort_values(by=sort_cols, inplace=True)

    # Remove line breaks and extra commas again for safety
    df.replace({r'[\r\n]+': ' ', ',': ' '}, regex=True, inplace=True)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # ----------------------------
    # ✅ EXPORT
    # ----------------------------
    try:
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        log(f"✅ Cleaned dataset saved: {OUTPUT_PATH}")
        log(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        log(f"❌ Failed to save cleaned file: {e}")
        raise

    log("=== NETFLIX ETL CLEAN FINISHED SUCCESSFULLY ===")

# ================================================================
# EXECUTION
# ================================================================
if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception:
        tb = traceback.format_exc()
        print("Unhandled exception:", tb, file=sys.stderr)
        sys.exit(1)
