
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


ARTIFACT_DIR = Path("streamlit_artifacts")

st.set_page_config(
    page_title="Recipe Rating Predictor",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1100px;
}

[data-testid="stSidebar"] {
    background-color: #fff7ed;
}

.hero {
    background: white;
    padding: 1.4rem 1.6rem;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    margin-bottom: 1.2rem;
}

.hero h1 {
    margin-bottom: 0.3rem;
    font-size: 2.4rem;
}

.hero p {
    color: #666;
    font-size: 1rem;
}

.rating-card {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    padding: 1.4rem;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 1.2rem;
}

.rating-number {
    font-size: 3.2rem;
    font-weight: 800;
    color: #c2410c;
}

.driver-card {
    background: white;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid #eee;
    margin-bottom: 0.7rem;
}
</style>
""", unsafe_allow_html=True)


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, k=1.5, enabled=True):
        self.k = k
        self.enabled = enabled

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.low_ = q1 - self.k * iqr
        self.high_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.enabled:
            return X
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


@st.cache_resource
def load_artifacts():
    model_pipeline = joblib.load(ARTIFACT_DIR / "rating_gradient_boosting_pipeline.pkl")
    raw_features = joblib.load(ARTIFACT_DIR / "rating_model_raw_features.pkl")
    processed_features = joblib.load(ARTIFACT_DIR / "rating_model_processed_features.pkl")
    background_raw = pd.read_csv(ARTIFACT_DIR / "shap_background_raw.csv")

    with open(ARTIFACT_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)

    return model_pipeline, raw_features, processed_features, background_raw, metadata


model_pipeline, raw_features, processed_features, background_raw, metadata = load_artifacts()

preprocess = model_pipeline.named_steps["preprocess"]
estimator = model_pipeline.named_steps["model"]


def make_safe_column_name(text, prefix="ing_"):
    text = str(text).lower().strip()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return prefix + text


def available_ingredient_names():
    names = []
    for col in raw_features:
        if col.startswith("ing_") and not col.startswith("inggrp_"):
            names.append(col.replace("ing_", "").replace("_", " "))
    return sorted(names)


def build_input_row(typed_ingredients, clicked_ingredients, calories, carbs, fat, protein):
    row = pd.DataFrame(0.0, index=[0], columns=raw_features)

    ingredients = []
    ingredients.extend([x.strip().lower() for x in typed_ingredients.split(",") if x.strip()])
    ingredients.extend([x.strip().lower() for x in clicked_ingredients])
    ingredients = sorted(set(ingredients))

    ingredient_count = len(ingredients)

    values = {
        "calories": calories,
        "carbs": carbs,
        "fat": fat,
        "protein": protein,
        "log_calories": np.log1p(max(calories, 0)),
        "log_carbs": np.log1p(max(carbs, 0)),
        "log_fat": np.log1p(max(fat, 0)),
        "log_protein": np.log1p(max(protein, 0)),
        "ingredient_count": ingredient_count,
        "log_ingredient_count": np.log1p(max(ingredient_count, 0)),
        "protein_per_calorie": protein / calories if calories else 0,
        "fat_per_calorie": fat / calories if calories else 0,
        "carbs_per_calorie": carbs / calories if calories else 0,
        "calories_per_ingredient": calories / ingredient_count if ingredient_count else 0,
    }

    for col, value in values.items():
        if col in row.columns:
            row.loc[0, col] = value

    for ingredient in ingredients:
        safe_col = make_safe_column_name(ingredient, prefix="ing_")
        if safe_col in row.columns:
            row.loc[0, safe_col] = 1.0

    return row, ingredients


def shap_explain(input_row):
    background_raw_aligned = background_raw.reindex(columns=raw_features, fill_value=0)
    input_row_aligned = input_row.reindex(columns=raw_features, fill_value=0)

    background_processed = preprocess.transform(background_raw_aligned)
    input_processed = preprocess.transform(input_row_aligned)

    explainer = shap.TreeExplainer(estimator, background_processed)
    shap_values = explainer.shap_values(input_processed)

    expected_value = float(np.ravel(explainer.expected_value)[0])
    shap_values_1d = np.ravel(shap_values)
    input_values_1d = np.ravel(input_processed)

    feature_clean_names = (
        pd.Series(processed_features)
        .str.replace("ing_", "", regex=False)
        .str.replace("inggrp_", "ingredient group: ", regex=False)
        .str.replace("_", " ", regex=False)
        .tolist()
    )

    shap_df = pd.DataFrame({
        "feature_clean": feature_clean_names,
        "value": input_values_1d,
        "shap_value": shap_values_1d,
    })

    shap_df = shap_df.reindex(
        shap_df["shap_value"].abs().sort_values(ascending=False).index
    )

    shap_explanation = shap.Explanation(
        values=shap_values_1d,
        base_values=expected_value,
        data=input_values_1d,
        feature_names=feature_clean_names
    )

    return shap_df, expected_value, shap_explanation


with st.sidebar:
    st.header("Recipe input")

    typed_ingredients = st.text_area(
        "Type ingredients separated by commas",
        value="chicken, garlic, onion, olive oil, tomato, salt, pepper"
    )

    clicked_ingredients = st.multiselect(
        "Click extra known ingredients",
        options=available_ingredient_names(),
        default=[]
    )

    st.markdown("---")
    st.subheader("Optional nutrition")

    calories = st.number_input("Calories", min_value=0.0, value=400.0)
    carbs = st.number_input("Carbs (g)", min_value=0.0, value=30.0)
    fat = st.number_input("Fat (g)", min_value=0.0, value=15.0)
    protein = st.number_input("Protein (g)", min_value=0.0, value=25.0)

    predict_clicked = st.button("Predict rating", use_container_width=True)


st.markdown("""
<div class="hero">
    <h1>🍽️ Recipe Rating Predictor</h1>
    <p>Enter ingredients to predict a recipe rating and see the main SHAP drivers.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Model: {metadata['champion_model_name']} | "
    f"MAE: {metadata['test_MAE']:.3f} | "
    f"RMSE: {metadata['test_RMSE']:.3f} | "
    f"R²: {metadata['test_R2']:.3f}"
)

if predict_clicked:
    input_row, ingredients = build_input_row(
        typed_ingredients=typed_ingredients,
        clicked_ingredients=clicked_ingredients,
        calories=calories,
        carbs=carbs,
        fat=fat,
        protein=protein,
    )

    prediction = float(model_pipeline.predict(input_row)[0])

    if "min_target_value" in metadata and "max_target_value" in metadata:
        prediction = min(max(prediction, metadata["min_target_value"]), metadata["max_target_value"])

    shap_df, expected_value, shap_explanation = shap_explain(input_row)

    st.markdown(f"""
    <div class="rating-card">
        <div>Predicted recipe rating</div>
        <div class="rating-number">{prediction:.2f} / 5</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Ingredients used:** " + (", ".join(ingredients) if ingredients else "None"))

    st.subheader("What is driving this rating?")

    fig = plt.figure(figsize=(8, 4.8))
    shap.plots.waterfall(shap_explanation, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)

    positive = shap_df[shap_df["shap_value"] > 0].head(5)
    negative = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value").head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Pushing rating up")
        for _, row in positive.iterrows():
            st.markdown(
                f"<div class='driver-card'>⬆️ <b>{row['feature_clean']}</b><br>"
                f"SHAP: {row['shap_value']:.3f}</div>",
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("#### Pushing rating down")
        for _, row in negative.iterrows():
            st.markdown(
                f"<div class='driver-card'>⬇️ <b>{row['feature_clean']}</b><br>"
                f"SHAP: {row['shap_value']:.3f}</div>",
                unsafe_allow_html=True
            )

else:
    st.info("Enter ingredients in the sidebar and click Predict rating.")
