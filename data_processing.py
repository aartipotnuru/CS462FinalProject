import json
import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RAW_PATH   = Path("data/raw/substitution_pairs.json")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────────────────────────────────────────
# DIETARY TAGS
# ─────────────────────────────────────────────────────────────────────────────
DIETARY_RULES = {
    "vegan":       ["egg", "milk", "butter", "cream", "honey", "cheese",
                    "yogurt", "meat", "chicken", "beef", "pork", "fish",
                    "shrimp", "gelatin", "lard", "ghee", "whey"],
    "gluten_free": ["flour", "wheat", "barley", "rye", "oats", "bread",
                    "pasta", "semolina", "spelt", "malt"],
    "nut_free":    ["almond", "peanut", "cashew", "walnut", "pecan",
                    "pistachio", "hazelnut", "macadamia", "nut butter", "tahini"],
    "dairy_free":  ["milk", "butter", "cream", "cheese", "yogurt",
                    "whey", "casein", "ghee", "lactose"],
    "keto":        ["sugar", "flour", "rice", "bread", "potato", "corn",
                    "oats", "pasta", "honey", "syrup", "molasses"],
}

def get_dietary_tags(ingredient: str) -> list[str]:
    """Return diets this ingredient is SAFE for (i.e. doesn't contain excludes)."""
    return [
        diet for diet, exclusions in DIETARY_RULES.items()
        if not any(ex in ingredient for ex in exclusions)
    ]

# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Lowercase, strip quantities/descriptors, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Remove quantities: "2 cups", "1/2 tsp", etc.
    text = re.sub(r'\d+[\./]?\d*\s*(cup|tbsp|tsp|oz|g|kg|ml|l|lb|pound|ounce)s?', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    # Remove common descriptors that don't affect substitution logic
    text = re.sub(
        r'\b(fresh|dried|chopped|sliced|minced|ground|large|small|medium|'
        r'unsifted|sifted|melted|softened|room temperature|all purpose|'
        r'all-purpose|seedless|dark|light|unsalted|salted)\b',
        '', text
    )
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(lemmatizer.lemmatize(w) for w in text.split() if w)
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
def load(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    with open(path) as f:
        data = json.load(f)

    # Keep only the fields we need
    df = pd.DataFrame(data)[["ingredient", "substitution"]]
    print(f"  Raw pairs:  {len(df):,}")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS
# ─────────────────────────────────────────────────────────────────────────────
def process(df: pd.DataFrame) -> pd.DataFrame:

    print("\n[1/4] Normalizing ingredients...")
    df["ingredient_clean"]   = df["ingredient"].apply(normalize)
    df["substitution_clean"] = df["substitution"].apply(normalize)

    # Drop anything that normalized to empty
    before = len(df)
    df = df[(df["ingredient_clean"] != "") & (df["substitution_clean"] != "")]
    # Drop self-substitutions (ingredient == its own substitute after cleaning)
    df = df[df["ingredient_clean"] != df["substitution_clean"]]
    # Drop exact duplicates
    df = df.drop_duplicates(subset=["ingredient_clean", "substitution_clean"])
    print(f"  Dropped {before - len(df):,} invalid/duplicate rows → {len(df):,} remain")

    print("\n[2/4] Tagging dietary categories...")
    df["ingredient_diets"]   = df["ingredient_clean"].apply(get_dietary_tags)
    df["substitution_diets"] = df["substitution_clean"].apply(get_dietary_tags)

    # Which diets does swapping to this substitute ENABLE?
    df["enables_diets"] = df.apply(
        lambda r: list(set(r["substitution_diets"]) - set(r["ingredient_diets"])),
        axis=1
    )

    print("\n[3/4] Building ingredient category labels...")
    # Simple rule-based category — useful feature for the model
    CATEGORIES = {
        "fat_oil":    ["butter", "oil", "lard", "ghee", "shortening", "margarine"],
        "flour":      ["flour", "starch", "cornstarch", "arrowroot", "semolina"],
        "dairy":      ["milk", "cream", "yogurt", "cheese", "buttermilk"],
        #"egg":        ["egg"],
        "sugar":      ["sugar", "honey", "syrup", "molasses", "sweetener"],
        "leavening":  ["baking soda", "baking powder", "yeast"],
        "liquid":     ["water", "broth", "stock", "juice", "wine", "beer"],
        "protein":    ["chicken", "beef", "pork", "fish", "tofu", "tempeh", "seitan"],
        #"herb_spice": ["pepper", "salt", "basil", "oregano", "thyme", "cumin"],
    }

    def categorize(ingredient: str) -> str:
        for cat, keywords in CATEGORIES.items():
            if any(kw in ingredient for kw in keywords):
                return cat
        return "other"

    df["ingredient_category"]   = df["ingredient_clean"].apply(categorize)
    df["substitution_category"] = df["substitution_clean"].apply(categorize)

    print("\n[4/4] Encoding categories...")
    le = LabelEncoder()
    all_cats = pd.concat([df["ingredient_category"], df["substitution_category"]])
    le.fit(all_cats)
    df["ingredient_category_id"]   = le.transform(df["ingredient_category"])
    df["substitution_category_id"] = le.transform(df["substitution_category"])

    with open(OUTPUT_DIR / "category_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# INGREDIENT INDEX  (used by the recommender in Part 3)
# ─────────────────────────────────────────────────────────────────────────────
def build_ingredient_index(df: pd.DataFrame):
    """Map every ingredient → list of its substitutes with metadata."""
    index = defaultdict(list)
    for _, row in df.iterrows():
        index[row["ingredient_clean"]].append({
            "substitute":          row["substitution_clean"],
            "enables_diets":       row["enables_diets"],
            "substitute_category": row["substitution_category"],
        })

    with open(OUTPUT_DIR / "ingredient_index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Saved ingredient_index.json — {len(index):,} ingredients")
    return index

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT & SAVE
# ─────────────────────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test   = train_test_split(temp, test_size=0.5, random_state=42)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}.csv"
        split.to_csv(path, index=False)
        print(f"  {name}: {len(split):,} rows → {path}")

    df.to_csv(OUTPUT_DIR / "full_processed.csv", index=False)

    summary = {
        "total_pairs":           len(df),
        "train":                 len(train),
        "val":                   len(val),
        "test":                  len(test),
        "unique_ingredients":    df["ingredient_clean"].nunique(),
        "unique_substitutions":  df["substitution_clean"].nunique(),
        "avg_subs_per_ingredient": round(
            df.groupby("ingredient_clean").size().mean(), 1
        ),
        "dietary_tags":          list(DIETARY_RULES.keys()),
        "ingredient_categories": list(df["ingredient_category"].unique()),
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Dataset summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Recipe Rewrite — Part 1: Data Processing ===\n")

    df = load(RAW_PATH)
    df = process(df)
    build_ingredient_index(df)

    print("\nSplitting and saving...")
    split_and_save(df)

    print("\n✅ Done. Outputs in data/processed/")
    print("   train.csv / val.csv / test.csv  ← for model training (Part 2)")
    print("   ingredient_index.json           ← for recommender (Part 3)")
    print("   category_encoder.pkl            ← for inference")
    print("   summary.json                    ← dataset stats")
