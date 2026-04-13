import sys
import os
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import torch
from utils import load_ingredient_index, load_category_encoder
from rec import rewrite_recipe, build_model

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH   = os.path.join(BASE_DIR, "model.pt")
INDEX_PATH   = os.path.join(BASE_DIR, "data", "processed", "ingredient_index.json")
ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "category_encoder.pkl")

index   = load_ingredient_index(INDEX_PATH)
encoder = load_category_encoder(ENCODER_PATH)
model   = build_model(len(encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def run(ingredients_text, restriction):
    cleaned_text = re.sub(r'\s*\([^)]*\)', '', ingredients_text)
    ingredients = [x.strip() for x in cleaned_text.split(",") if x.strip()]
    if not ingredients:
        return "Please enter at least one ingredient.", ""

    new_recipe, replacements = rewrite_recipe(
        ingredients, restriction, model, encoder, index
    )

    if not replacements:
        replacements_text = "No replacements needed — all ingredients already fit!"
    else:
        replacements_text = ""
        for original, subs in replacements.items():
            replacements_text += f"{original}  →  {', '.join(subs)}\n"

    new_recipe_text = ", ".join(new_recipe)
    return replacements_text.strip(), new_recipe_text

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Textbox(label="Ingredients (comma-separated)", placeholder="e.g. milk, butter, eggs, flour"),
        gr.Dropdown(
            choices=["vegan", "vegetarian", "gluten_free", "dairy_free", "nut_free", "keto"],
            label="Dietary restriction",
            value="vegan"
        )
    ],
    outputs=[
        gr.Textbox(label="Replacements"),
        gr.Textbox(label="New ingredient list")
    ],
    title="Recipe Rewriter",
    description="Enter your ingredients and a dietary restriction to get substitutions.",
)

if __name__ == "__main__":
    demo.launch()