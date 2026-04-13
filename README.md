# CS462FinalProject
Recipe Rewrite - A Deep Learning Approach to Ingredient Substitution using Python Machine Learning Models

Problem: Many people who have several dietary restictions, allergies, or different choices of eating (such as a diet) can find it challenging at times to find ingredient substitutions for cooking recipes.

Approach: We will be using Python-based deep learning models such as TensorFlow and PyTorch to train datasets and recieve alternatives while still maintaining the quality of the recipe.

Goal: The models serve as a purpose to provide accurate alternatives for users to make cooking more accessible and convenient.

Project Presentation Slides: [https://docs.google.com/presentation/d/1VQR533iFUqO5bZmVxefhmozNDlNKKxLjwx0ZRuj6iJs/edit?usp=sharing](https://docs.google.com/presentation/d/1VQR533iFUqO5bZmVxefhmozNDlNKKxLjwx0ZRuj6iJs/edit?usp=sharing)

Demo: https://www.youtube.com/watch?v=WcuuFNeQ1zQ
## Overview

This project builds a recipe substitution assistant that helps users replace ingredients based on dietary restrictions like vegan, gluten-free, dairy-free, nut-free, and keto.

The main workflow is:
1. Process raw substitution data with `data_processing.py`
2. Train the model with `deepLearningModel/trainPyTorch.py`
3. Launch the Gradio web interface with `recommenderSystem/app_gradio.py`

## Prerequisites

- Python 3.8+ (recommended)
- A virtual environment is strongly recommended
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `nltk`
  - `torch`
  - `gradio`

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn nltk torch gradio
```

> `data_processing.py` downloads the `wordnet` package automatically when it runs.

## Step 1: Process the data

Run the preprocessing step first so the raw dataset is converted into training, validation, and test CSV files.

```bash
python data_processing.py
```

This creates files under `data/processed/`:

- `train.csv`
- `val.csv`
- `test.csv`
- `full_processed.csv`
- `ingredient_index.json`
- `category_encoder.pkl`
- `summary.json`

If this step fails, make sure the raw dataset exists at `data/raw/substitution_pairs.json`.

## Step 2: Train the model

Run the PyTorch training script next.

```bash
python deepLearningModel/trainPyTorch.py
```

This script loads:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

It trains a classifier over ingredient categories and prints validation/test accuracy.

## Step 3: Launch the Gradio app

Start the recommendation interface from the project root:

```bash
python recommenderSystem/app_gradio.py
```

Then open the local Gradio link shown in the terminal.

The app accepts:

- comma-separated ingredients
- a dietary restriction from:
  - `vegan`
  - `vegetarian`
  - `gluten_free`
  - `dairy_free`
  - `nut_free`
  - `keto`

And it returns suggested ingredient replacements plus the rewritten ingredient list.

## Project files to know

- `data_processing.py` — raw data normalization, dietary tag extraction, and split generation
- `deepLearningModel/trainPyTorch.py` — model training and evaluation
- `recommenderSystem/app_gradio.py` — Gradio frontend for recipe rewriting
- `recommenderSystem/rec.py` — recommendation logic and substitution ranking
- `recommenderSystem/utils.py` — utility loaders and ingredient normalization

## Running the full flow

1. `python data_processing.py`
2. `python deepLearningModel/trainPyTorch.py`
3. `python recommenderSystem/app_gradio.py`
