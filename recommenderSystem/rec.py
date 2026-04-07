import torch
import torch.nn as nn

from utils import (
    load_ingredient_index,
    load_category_encoder,
    normalize_ingredient,
    filter_by_diet,
    already_satisfies
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
PROTEIN_WORDS = [
    "tofu", "bean", "lentil", "chickpea", "seitan",
    "turkey", "beef", "pork", "chicken", "fish",
    "shrimp", "meat"
]

GOOD_PATTERNS = {
    "milk": ["milk", "cream", "oat", "soy", "almond"],
    "chicken": ["tofu", "bean", "lentil", "chickpea"]
}

BAD_CONTEXT_WORDS = [
    "coffee", "water", "salt", "seasoning", "spice",
    "extract", "powder", "sauce", "baking soda"
]

# ─────────────────────────────────────────────────────────────
# SIMPLE SIMILARITY 
# ─────────────────────────────────────────────────────────────
def similarity(a, b):
    a_words = set(a.split())
    b_words = set(b.split())
    return len(a_words & b_words)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# RECOMMENDER
# ─────────────────────────────────────────────────────────────
def recommend(ingredient, restriction, model, encoder, index):
    ingredient = normalize_ingredient(ingredient)

    if ingredient not in index:
        return []

    candidates = index[ingredient]
    candidates = filter_by_diet(candidates, restriction)
    if not candidates:
        return []

    pred_category = predict_category(model, encoder, ingredient)

    def is_protein(name):
        return any(word in name for word in PROTEIN_WORDS)

    # Optional protein filtering 
    if any(word in ingredient for word in PROTEIN_WORDS):
        protein_filtered = [
            c for c in candidates
            if c["substitute_category"] == "protein" or is_protein(c["substitute"])
        ]
        if protein_filtered:
            candidates = protein_filtered

    # ─────────────────────────────────────────────────────────
    # SCORING FUNCTION 
    # ─────────────────────────────────────────────────────────
    def score(c):
        s = 0
        sub = c["substitute"]

        # 1. Category match (Fix #3: soft model usage)
        if pred_category:
            if c["substitute_category"] == pred_category:
                s += 5
            else:
                s -= 1

        # 2. Protein consistency
        if any(word in ingredient for word in PROTEIN_WORDS):
            if any(word in sub for word in PROTEIN_WORDS):
                s += 3

        # 3. Similarity boost (Fix #2)
        s += 2 * similarity(ingredient, sub)

        # 4. Learned pattern boost (Fix #4)
        for key, words in GOOD_PATTERNS.items():
            if key in ingredient:
                if any(w in sub for w in words):
                    s += 3

        # 5. Light penalties (NOT hard filtering)
        if any(word in sub for word in BAD_CONTEXT_WORDS):
            s -= 3

        if len(sub.split()) > 2:
            s -= 1

        return s

    ranked = sorted(candidates, key=score, reverse=True)

    return [c["substitute"] for c in ranked[:3]]

# ─────────────────────────────────────────────────────────────
# REWRITE PIPELINE
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
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