import json
import pickle

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data_processing import normalize

def load_ingredient_index(path="data/processed/ingredient_index.json"):
    with open(path) as f:
        return json.load(f)


def load_category_encoder(path="data/processed/category_encoder.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def normalize_ingredient(x):
    return normalize(x)

def filter_by_diet(candidates, restriction):
    restriction = restriction.lower().strip()

    filtered = []

    for c in candidates:
        diets = c.get("enables_diets", [])
        name = c["substitute"]

        # dataset check
        if restriction not in diets:
            continue

        # fix bad labels
        if restriction == "vegan":
            if any(x in name for x in ["milk", "cream", "cheese", "butter", "yogurt", "half and half"]):
                continue

        if restriction == "dairy_free":
            if any(x in name for x in ["milk", "cream", "cheese", "butter", "yogurt"]):
                continue

        filtered.append(c)

    return filtered

def already_satisfies(ingredient, restriction, index):
    if ingredient not in index:
        return False

    entries = index[ingredient]
    diets = entries[0].get("ingredient_diets", [])

    return restriction in diets