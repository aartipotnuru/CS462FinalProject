import torch
import torch.nn as nn

from utils import (
    load_ingredient_index,
    load_category_encoder,
    normalize_ingredient,
    filter_by_diet,
    already_satisfies
)

PROTEIN_WORDS = [
    "tofu", "bean", "lentil", "chickpea", "seitan",
    "turkey", "beef", "pork", "chicken", "fish",
    "shrimp", "meat"
]

INVALID_TYPES = [
    "baking soda", "baking powder", "yeast",
    "water", "salt", "seasoning", "spice",
    "extract", "powder"
]

def build_model(num_classes):
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(128, 64),
        nn.ReLU(),

        nn.Linear(64, num_classes)
    )


def load_model(path, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def predict_category(model, encoder, ingredient):
    try:
        cat = encoder.transform([ingredient])[0]
    except:
        return None

    x = torch.tensor([[cat]], dtype=torch.float32)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return encoder.inverse_transform([pred])[0]

def recommend(ingredient, restriction, model, encoder, index):
    ingredient = normalize_ingredient(ingredient)

    if ingredient not in index:
        return []

    candidates = index[ingredient]
    candidates = filter_by_diet(candidates, restriction)
    if not candidates:
        return []

    candidates = [
        c for c in candidates
        if not any(bad in c["substitute"] for bad in INVALID_TYPES)
    ]

    if not candidates:
        return []

    pred_category = predict_category(model, encoder, ingredient)

    def is_protein(name):
        return any(word in name for word in PROTEIN_WORDS)

    if any(word in ingredient for word in PROTEIN_WORDS):
        protein_filtered = [
            c for c in candidates
            if c["substitute_category"] == "protein" or is_protein(c["substitute"])
        ]
        if protein_filtered:
            candidates = protein_filtered

    if pred_category:
        model_filtered = [
            c for c in candidates
            if c["substitute_category"] == pred_category
        ]
        if model_filtered:
            candidates = model_filtered

    def score(c):
        s = 0

        if pred_category and c["substitute_category"] == pred_category:
            s += 3

        if any(word in c["substitute"] for word in PROTEIN_WORDS):
            s += 1

        return s

    ranked = sorted(candidates, key=score, reverse=True)

    return [c["substitute"] for c in ranked[:3]]

def rewrite_recipe(ingredients, restriction, model, encoder, index):
    new_recipe = []
    replacements = {}

    restriction = restriction.lower().strip()

    for ing in ingredients:
        ing_clean = normalize_ingredient(ing)

        if already_satisfies(ing_clean, restriction, index):
            new_recipe.append(ing_clean)
            continue

        subs = recommend(ing_clean, restriction, model, encoder, index)

        if subs:
            new_recipe.append(subs[0])
            replacements[ing] = subs
        else:
            new_recipe.append(ing)

    return new_recipe, replacements


def main():
    print("=== MODEL-BASED RECIPE REWRITER ===")

    MODEL_PATH = "model.pt"

    index = load_ingredient_index()
    encoder = load_category_encoder()

    model = load_model(MODEL_PATH, len(encoder.classes_))

    recipe_input = input("Enter ingredients: ")
    restriction = input("Enter restriction (vegan, vegetarian, gluten_free, nut_free, dairy_free, keto): ")

    ingredients = [x.strip() for x in recipe_input.split(",")]

    new_recipe, replacements = rewrite_recipe(
        ingredients, restriction, model, encoder, index
    )

    print("\nNew recipe:", new_recipe)
    print("\nReplacements:")

    if not replacements:
        print("No replacements needed!")
    else:
        for k, v in replacements.items():
            print(f"{k} → {v}")


if __name__ == "__main__":
    main()