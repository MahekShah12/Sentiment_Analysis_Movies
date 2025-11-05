import pandas as pd
import numpy as np
import os
import re

# ===============================
# 🔹 CONFIGURATION
# ===============================
INPUT_CSV = r"C:\Users\mahek\Desktop\DATASETS_2025\box_office_data.csv"
OUTPUT_CSV = r"C:\Users\mahek\Desktop\DATASETS_2025\cleaned_box_office.csv"


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

# Columns to keep (matching actual dataset)
columns_to_keep = [
    "Rank", "Release Group", "$Worldwide",
    "Domestic %", "Foreign %", "Year",
    "Genres", "Rating", "Production_Countries"
]

# Verify available columns
available_cols = [col for col in columns_to_keep if col in df.columns]
missing = set(columns_to_keep) - set(available_cols)
if missing:
    print(f"⚠️ Missing columns in CSV: {missing}")
df = df[available_cols].copy()

# Standardize column names
df.rename(columns={
    "Rank": "rank",
    "Release Group": "release_title",
    "$Worldwide": "worldwide_gross",
    "Domestic %": "domestic_percent",
    "Foreign %": "foreign_percent",
    "Year": "year",
    "Genres": "genre",
    "Rating": "rating",
    "Production_Countries": "production_countries"
}, inplace=True)

# ---------------------------------
# 🧼 DATA CLEANING
# ---------------------------------

# Drop rows missing essential fields
df.dropna(subset=["release_title", "year"], inplace=True)

# Clean numeric values (remove $ and %)
def clean_currency(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '').replace(',', '').strip()
    return float(val) if val.replace('.', '', 1).isdigit() else np.nan

df["worldwide_gross"] = df["worldwide_gross"].apply(clean_currency)
df["domestic_percent"] = df["domestic_percent"].astype(str).str.replace('%', '').astype(float)
df["foreign_percent"] = df["foreign_percent"].astype(str).str.replace('%', '').astype(float)

# Handle missing numeric values
df["worldwide_gross"].fillna(df["worldwide_gross"].median(), inplace=True)
df["domestic_percent"].fillna(0, inplace=True)
df["foreign_percent"].fillna(0, inplace=True)

# Convert year to numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

# ---------------------------------
# NLP-Style Text Cleaning
# ---------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text_cols = ["release_title", "genre", "rating", "production_countries"]
for col in text_cols:
    df[col] = df[col].apply(clean_text)

# Drop duplicates (same movie same year)
df.drop_duplicates(subset=["release_title", "year"], keep="first", inplace=True)

# Drop columns that are almost empty (<20% filled)
threshold = len(df) * 0.2
df.dropna(axis=1, thresh=threshold, inplace=True)

# Fill missing text columns
df.fillna({
    "genre": "unknown",
    "rating": "unrated",
    "production_countries": "unspecified"
}, inplace=True)

# Reorder columns
final_df = df[[
    "rank", "release_title", "worldwide_gross",
    "domestic_percent", "foreign_percent",
    "year", "genre", "rating", "production_countries"
]]

print(f"✅ Cleaned dataset shape: {final_df.shape}")

# ===============================
# 💾 3️⃣ LOAD
# ===============================
print("💾 Saving cleaned CSV...")

final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ File saved successfully at:\n{OUTPUT_CSV}")
