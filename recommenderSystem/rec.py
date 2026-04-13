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
    "egg": ["flax", "flax seed", "chia", "applesauce", "silken tofu"],
    "milk": ["milk", "cream", "oat", "soy", "almond", "rice", "coconut"],
    "cream": ["cream", "coconut cream", "cashew cream", "oat cream", "soy cream", "silken tofu"],
    "chicken": ["tofu", "bean", "lentil", "chickpea"],
    "peanut": ["seed", "sunflower", "pumpkin", "soy", "butter"],
    "almond milk": ["milk", "oat", "rice", "soy", "coconut"],
    "cheese": ["nutritional yeast", "miso", "tofu", "vegan cheese"]
}

BAD_CONTEXT_WORDS = [
    "coffee", "water", "salt", "seasoning", "spice",
    "extract", "powder", "sauce", "baking soda",
    "cilantro", "chocolate chip", "ranch dressing"
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
# REPLACE WITH this (matches trainPyTorch.py exactly):
EMBEDDING_SIZE = 8
NUM_CATEGORIES = 10
DIET_AMT = 5

class Model(nn.Module):
    def __init__(self, count, size, dietAmt, tot):
        super(Model, self).__init__()
        self.embed = nn.Embedding(tot, size)
        response = size + dietAmt
        self.layering = nn.Sequential(
            nn.Linear(response, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, tot)
        )

    def forward(self, first, second):
        amtOne = self.embed(first.view(-1))        # shape [batch]  → [batch, embed_size]
        amtTwo = torch.cat([amtOne, second], dim=1)
        return self.layering(amtTwo)

def build_model(num_classes):
    return Model(
        count=NUM_CATEGORIES,
        size=EMBEDDING_SIZE,
        dietAmt=DIET_AMT,
        tot=num_classes
    )

def load_model(path, num_classes):
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict_category(model, encoder, ingredient):
    try:
        cat = encoder.transform([ingredient])[0]
    except:
        return None

    x = torch.tensor([cat], dtype=torch.long).unsqueeze(0)   # shape [1, 1]
    diet = torch.zeros(1, DIET_AMT, dtype=torch.float32)

    with torch.no_grad():
        out = model(x, diet)
        pred = torch.argmax(out, dim=1).item()

    return encoder.inverse_transform([pred])[0]

# ─────────────────────────────────────────────────────────────
# RECOMMENDER
# ─────────────────────────────────────────────────────────────
def recommend(ingredient, restriction, model, encoder, index):
    if ingredient not in index:
        ingredient = normalize_ingredient(ingredient)

    if ingredient not in index:
        return []

    candidates = index[ingredient]
    candidates = filter_by_diet(candidates, restriction)
    if not candidates:
        return []

    pred_category = predict_category(model, encoder, ingredient)

    def is_protein(name):
        name = name.lower()
        return any(word in name for word in PROTEIN_WORDS)

    def is_milk_like(name):
        name = name.lower()
        milk_words = ["milk", "oat", "soy", "almond", "rice", "coconut", "cream"]
        return any(word in name for word in milk_words)

    def is_nut_or_seed_like(name):
        name = name.lower()
        words = [
            "seed", "sunflower", "pumpkin", "sesame",
            "soy", "butter", "spread", "pea"
        ]
        return any(word in name for word in words)

    def is_cream_like(name):
        name = name.lower()
        cream_words = [
            "cream", "coconut cream", "cashew cream",
            "soy cream", "oat cream", "silken tofu"
        ]
        return any(word in name for word in cream_words)

    def is_egg_like(name):
        name = name.lower()
        egg_words = [
            "flax", "chia", "applesauce", "silken tofu",
            "egg replacer", "aquafaba"
        ]
        return any(word in name for word in egg_words)

    # protein ingredients should stay protein-like
    if any(word in ingredient for word in PROTEIN_WORDS):
        protein_filtered = [
            c for c in candidates
            if c["substitute_category"] == "protein" or is_protein(c["substitute"])
        ]
        if protein_filtered:
            candidates = protein_filtered

    # milk ingredients should stay milk-like
    if "milk" in ingredient:
        milk_filtered = [
            c for c in candidates
            if is_milk_like(c["substitute"])
        ]
        if milk_filtered:
            candidates = milk_filtered

    # cream ingredients should stay cream-like
    if "cream" in ingredient:
        cream_filtered = [
            c for c in candidates
            if is_cream_like(c["substitute"])
        ]
        if cream_filtered:
            candidates = cream_filtered

    # egg ingredients should stay egg-replacer-like
    if "egg" in ingredient:
        egg_filtered = [
            c for c in candidates
            if is_egg_like(c["substitute"])
        ]
        if egg_filtered:
            candidates = egg_filtered

    # peanut/nut ingredients should prefer seed/spread-like replacements
    if any(word in ingredient for word in ["peanut", "nut", "almond", "cashew"]):
        nut_filtered = [
            c for c in candidates
            if is_nut_or_seed_like(c["substitute"])
        ]
        if nut_filtered:
            candidates = nut_filtered

    def score(c):
        s = 0
        sub = c["substitute"].lower()

        # 1. soft category match
        if pred_category:
            if c["substitute_category"] == pred_category:
                s += 5
            else:
                s -= 1

        # 2. protein consistency
        if any(word in ingredient for word in PROTEIN_WORDS):
            if any(word in sub for word in PROTEIN_WORDS):
                s += 3

        # 3. token overlap
        s += 2 * similarity(ingredient, sub)

        # 4. learned ingredient patterns
        for key, words in GOOD_PATTERNS.items():
            if key in ingredient:
                if any(w in sub for w in words):
                    s += 4

        # 5. structure boosts
        if "milk" in ingredient and is_milk_like(sub):
            s += 4

        if "cream" in ingredient and is_cream_like(sub):
            s += 5

        if "egg" in ingredient and is_egg_like(sub):
            s += 5

        if any(word in ingredient for word in ["peanut", "nut", "almond", "cashew"]) and is_nut_or_seed_like(sub):
            s += 4

        # 6. cream-specific penalty
        if "cream" in ingredient:
            if any(word in sub for word in ["yogurt", "yoghurt", "ranch", "feta"]):
                s -= 6

        # 7. generic-word penalty
        if sub in ["fat", "liquid", "powder", "spread"]:
            s -= 5

        # 8. weird pantry/context penalties
        if any(word in sub for word in BAD_CONTEXT_WORDS):
            s -= 4

        if len(sub.split()) > 3:
            s -= 1

        return s

    ranked = sorted(candidates, key=score, reverse=True)
    return [c["substitute"] for c in ranked[:3]]

# ─────────────────────────────────────────────────────────────
# REWRITE PIPELINE
# ─────────────────────────────────────────────────────────────
NUT_LOOKUP_MAP = {
    "cashew milk": "milk",
    "cashew cream": "cream",
    "almond cream": "cream",
    "almond milk": "milk",
}

def rewrite_recipe(ingredients, restriction, model, encoder, index):
    new_recipe = []
    replacements = {}

    restriction = restriction.lower().strip()

    for ing in ingredients:
        ing_clean = normalize_ingredient(ing)

        lookup_key = NUT_LOOKUP_MAP.get(ing_clean, ing_clean)
        force_replace = (lookup_key != ing_clean and restriction == "nut_free")

        if not force_replace and already_satisfies(ing_clean, restriction, index):
            new_recipe.append(ing_clean)
            continue

        subs = recommend(lookup_key, restriction, model, encoder, index)

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