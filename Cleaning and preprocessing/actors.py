import pandas as pd
import json
import re
from tqdm import tqdm

# --- CONFIG ---
input_file = r"C:\Users\mahek\Desktop\DATASETS_2025\actors.json"
output_file = r"C:\Users\mahek\Desktop\DATASETS_2025\cleaned_actors.csv"

# --- Step 1️⃣ Extract: Load JSONL file efficiently ---
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading actor JSON lines"):
        try:
            record = json.loads(line.strip())
            data.append(record)
        except json.JSONDecodeError:
            continue  # skip malformed lines

df = pd.DataFrame(data)
print("✅ JSON Loaded Successfully!")
print("📊 Shape:", df.shape)

# --- Step 2️⃣ Transform: Keep only required columns ---
keep_cols = ["actor_id", "name", "birthYear"]
df = df[keep_cols].copy()

# --- Step 3️⃣ Cleaning Function ---
def clean_text(text):
    """Remove unwanted characters, commas, and non-ASCII symbols."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)       # remove newlines/tabs
    text = re.sub(r"\s+", " ", text)             # collapse multiple spaces
    text = re.sub(r"[\"\'\[\]\{\}]", "", text)   # remove quotes/brackets
    text = text.replace(",", " ")                # replace commas with spaces
    text = re.sub(r"[^\x00-\x7F]+", "", text)    # remove special/non-ASCII chars
    return text.strip()

# Apply cleaning to all columns
for col in ["actor_id", "name", "birthYear"]:
    df[col] = df[col].apply(clean_text)

# --- Step 4️⃣ Normalize data types ---
df["birthYear"] = pd.to_numeric(df["birthYear"], errors="coerce").astype("Int64")

# --- Step 5️⃣ Final cleanup ---
df.drop_duplicates(subset=["actor_id"], inplace=True)
df.dropna(subset=["actor_id", "name"], inplace=True)

# --- Step 6️⃣ Reorder columns ---
df = df[["actor_id", "name", "birthYear"]]

# --- Step 7️⃣ Save output ---
df.to_csv(output_file, index=False, encoding="utf-8")

print("\n✅ ETL Completed Successfully!")
print("📁 Output Saved To:", output_file)
print("📊 Final Shape:", df.shape)
print("📄 Columns:", list(df.columns))

