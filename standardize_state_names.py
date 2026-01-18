import pandas as pd
import os

# normalize
def normalize_state(col):
    return (
        col.astype(str)
        .str.lower()
        .str.strip()
        .str.replace("&", "and")
        .str.replace(r"\s+", " ", regex=True)
    )

# remove invalid states
def clean_invalid_states(col):

    col = col.replace(r"^\d+$", "unknown", regex=True)

    
    col = col.replace(["", "nan", "none", "null"], "unknown")
    return col


STATE_MAP = {
    # historical names
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",

    # Andaman & Nicobar
    "andaman and nicobar islands": "andaman & nicobar islands",

    # Dadra & Nagar Haveli 
    "dadra and nagar haveli and daman and diu":
        "dadra and nagar haveli and daman & diu",
    "the dadra and nagar haveli and daman and diu":
        "dadra and nagar haveli and daman & diu",

    # West Bengal fixes
    "west bangal": "west bengal",
    "westbengal": "west bengal"
}

# find data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

def standardize_file(filepath):
    df = pd.read_csv(filepath)

    if "state" not in df.columns:
        print(f" Skipping (no state column): {os.path.basename(filepath)}")
        return

    
    df["state"] = normalize_state(df["state"])
    df["state"] = clean_invalid_states(df["state"])
    df["state"] = df["state"].replace(STATE_MAP)

    
    df.to_csv(filepath, index=False)
    print(f" Fixed & saved: {os.path.basename(filepath)}")


def main():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f" Data folder not found: {DATA_DIR}")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            standardize_file(os.path.join(DATA_DIR, file))


if __name__ == "__main__":
    main()
