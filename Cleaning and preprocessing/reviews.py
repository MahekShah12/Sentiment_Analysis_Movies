import pandas as pd
import re
import os

# === CONFIG ===
INPUT_FILE = r"C:\Users\mahek\Desktop\DATASETS_2025\tmdb_reviews_with_sentiment.csv"
OUTPUT_FILE = r"C:\Users\mahek\Desktop\DATASETS_2025\tmdb_cleaned.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# === 1️⃣ EXTRACT ===
print("📥 Loading dataset...")
df = pd.read_csv(INPUT_FILE, encoding='utf-8')
print(f"✅ Loaded! Shape: {df.shape}")

# === 2️⃣ CLEAN TEXT FUNCTION ===
def clean_review(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()                       # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)    # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)          # keep only letters and spaces
    text = re.sub(r"\s+", " ", text)               # collapse extra spaces
    return text.strip()

# === 3️⃣ APPLY CLEANING ON 'review' COLUMN ===
if "review" in df.columns:
    print("🧹 Cleaning review text...")
    df["review"] = df["review"].astype(str).apply(clean_review)

# === 4️⃣ KEEP ONLY ONE CLEAN REVIEW COLUMN ===
# If multiple columns like 'review_x', 'review_y', etc. exist — merge and keep only one
review_cols = [col for col in df.columns if "review" in col.lower()]
if len(review_cols) > 1:
    print(f"⚙️ Found multiple review columns: {review_cols}. Keeping the cleaned one only.")
    df["review"] = df[review_cols[0]]  # Keep the first valid cleaned one
    df.drop(columns=[c for c in review_cols if c != "review"], inplace=True)

# === 5️⃣ CLEAN OTHER TEXT COLUMNS (trim spaces) ===
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

# === 6️⃣ DROP EMPTY REVIEWS ===
if "review" in df.columns:
    df = df[df["review"].str.len() > 0]

# === 7️⃣ SAVE CLEANED DATA ===
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"✅ Cleaned dataset saved at:\n{OUTPUT_FILE}")
print(f"📊 Final shape: {df.shape}")
