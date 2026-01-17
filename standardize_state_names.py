import pandas as pd
import os

# ---------- NORMALIZE STATE TEXT ----------
def normalize_state(col):
    return (
        col.astype(str)
        .str.lower()
        .str.strip()
        .str.replace("&", "and")
        .str.replace(r"\s+", " ", regex=True)
    )

# ---------- REMOVE INVALID / NUMERIC STATES ----------
def clean_invalid_states(col):
    # replace pure numbers like 100000 with 'unknown'
    col = col.replace(r"^\d+$", "unknown", regex=True)

    # replace empty / nan-like values
    col = col.replace(["", "nan", "none", "null"], "unknown")
    return col

# ---------- CANONICAL STATE NAMES ----------
STATE_MAP = {
    # Historical names
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",

    # Andaman & Nicobar
    "andaman and nicobar islands": "andaman & nicobar islands",

    # Dadra & Nagar Haveli (ALL variants → one official name)
    "dadra and nagar haveli and daman and diu":
        "dadra and nagar haveli and daman & diu",
    "the dadra and nagar haveli and daman and diu":
        "dadra and nagar haveli and daman & diu",

    # West Bengal fixes
    "west bangal": "west bengal",
    "westbengal": "west bengal"
}

# ---------- FIND DATA FOLDER SAFELY ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def standardize_file(filepath):
    df = pd.read_csv(filepath)

    if "state" not in df.columns:
        print(f"⚠ Skipping (no state column): {os.path.basename(filepath)}")
        return

    # Normalize → clean → map
    df["state"] = normalize_state(df["state"])
    df["state"] = clean_invalid_states(df["state"])
    df["state"] = df["state"].replace(STATE_MAP)

    # overwrite SAME file
    df.to_csv(filepath, index=False)
    print(f"✅ Fixed & saved: {os.path.basename(filepath)}")


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ Data folder not found: {DATA_DIR}")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            standardize_file(os.path.join(DATA_DIR, file))


if __name__ == "__main__":
    main()
