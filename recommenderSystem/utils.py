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
    x = normalize(x)

    SPECIAL_CASES = {
        "whole milk": "milk",
        "skim milk": "milk",
        "2% milk": "milk",
        "full fat milk": "milk",

        "eggs": "egg",
        "egg yolks": "egg",
        "egg whites": "egg",

        "peanuts": "peanut",
        "chopped peanuts": "peanut",
        "roasted peanuts": "peanut",

        "butters": "butter",
    }

    return SPECIAL_CASES.get(x, x)

def filter_by_diet(candidates, restriction):
    restriction = restriction.lower().strip()
    filtered = []

    for c in candidates:
        diets = c.get("enables_diets", [])
        name = c["substitute"].lower().strip()

        # must pass dataset diet tag first
        if restriction not in diets:
            continue

        # manual cleanup for mislabeled dataset entries
        if restriction == "vegan":
            banned_words = [
                "milk", "cream", "cheese", "butter", "yogurt", "yoghurt",
                "half and half", "feta", "mozzarella", "parmesan", "cheddar",
                "cream cheese", "ricotta", "brie", "goat cheese",
                "egg", "yolk", "white", "mayonnaise", "mayo", "aioli", "custard",
                "honey", "gelatin", "whey", "casein", "buttermilk", "sour cream",
                "ice cream", "ranch"
            ]
            if any(x in name for x in banned_words):
                continue

        if restriction == "dairy_free":
            banned_words = [
                "milk", "cream", "cheese", "butter", "yogurt", "yoghurt",
                "half and half", "feta", "mozzarella", "parmesan", "cheddar",
                "cream cheese", "ricotta", "brie", "goat cheese",
                "buttermilk", "whey", "casein", "sour cream", "ice cream", "ranch"
            ]
            if any(x in name for x in banned_words):
                continue

        if restriction == "vegetarian":
            banned_words = [
                "chicken", "beef", "pork", "turkey", "fish", "shrimp", "meat",
                "bacon", "ham", "anchovy"
            ]
            if any(x in name for x in banned_words):
                continue

        if restriction == "nut_free":
            banned_words = [
                "almond", "cashew", "walnut", "pecan", "hazelnut", "pistachio",
                "peanut", "mixed nut"
            ]
            if any(x in name for x in banned_words):
                continue

        if restriction == "gluten_free":
            banned_words = [
                "bread", "flour", "pasta", "wheat", "barley", "rye", "couscous",
                "spaetzle", "couscou"
            ]
            if any(x in name for x in banned_words):
                continue

        filtered.append(c)

    return filtered

def already_satisfies(ingredient, restriction, index):
    if ingredient not in index:
        return False

    entries = index[ingredient]
    diets = entries[0].get("ingredient_diets", [])

    return restriction in diets