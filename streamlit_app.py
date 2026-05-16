
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------
# App setup
# --------------------------------------------------

ARTIFACT_DIR = Path("streamlit_artifacts")

st.set_page_config(
    page_title="Recipe Rating Predictor",
    page_icon="🍽️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #faf7f2;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #3b2f2f;
    font-size: 3rem;
    font-weight: 800;
}

h2, h3 {
    color: #4a3b35;
}

[data-testid="stSidebar"] {
    background-color: #fff4e6;
}

.recipe-card {
    background-color: white;
    padding: 1.5rem;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #fff7ed, #ffedd5);
    padding: 1.5rem;
    border-radius: 20px;
    border: 1px solid #fed7aa;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    text-align: center;
}

.big-rating {
    font-size: 3rem;
    font-weight: 800;
    color: #c2410c;
}

.small-muted {
    color: #6b7280;
    font-size: 0.95rem;
}

.positive-box {
    background-color: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 1rem;
    border-radius: 14px;
}

.negative-box {
    background-color: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 1rem;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Custom transformer needed to load saved pipeline
# --------------------------------------------------

class IQRClipper(BaseEstimator, TransformerMixin):
    """IQR winsorizer used inside the saved preprocessing pipeline."""

    def __init__(self, k=1.5, enabled=True):
        self.k = k
        self.enabled = enabled

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        low = q1 - self.k * iqr
        high = q3 + self.k * iqr
        self.low_ = np.where(np.isfinite(low), low, -np.inf)
        self.high_ = np.where(np.isfinite(high), high, np.inf)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if not self.enabled:
            return X
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------

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


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def make_safe_column_name(text, prefix="ing_"):
    text = str(text).lower().strip()
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return prefix + text


def available_ingredient_names():
    names = []

    for col in raw_features:
        if col.startswith("ing_") and not col.startswith("inggrp_"):
            clean_name = col.replace("ing_", "").replace("_", " ")
            names.append(clean_name)

    return sorted(names)


def build_input_row(typed_ingredients, clicked_ingredients, calories, carbs, fat, protein):
    # Important: use 0.0 so pandas allows decimal values
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

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

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
        "feature": processed_features,
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


# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

with st.sidebar:
    st.header("🍳 Recipe input")

    typed_ingredients = st.text_area(
        "Type ingredients separated by commas",
        value="chicken, garlic, onion, olive oil, tomato, salt, pepper"
    )

    ingredient_options = available_ingredient_names()

    clicked_ingredients = st.multiselect(
        "Or click ingredients from the model's known ingredient list",
        options=ingredient_options,
        default=[]
    )

    st.subheader("🥦 Optional nutrition inputs")

    calories = st.number_input("Calories", min_value=0.0, value=400.0)
    carbs = st.number_input("Carbs (g)", min_value=0.0, value=30.0)
    fat = st.number_input("Fat (g)", min_value=0.0, value=15.0)
    protein = st.number_input("Protein (g)", min_value=0.0, value=25.0)

    predict_clicked = st.button("Predict rating")


# --------------------------------------------------
# Main app layout
# --------------------------------------------------

st.markdown("""
<div class="recipe-card">
    <h1>🍽️ Recipe Rating Predictor</h1>
    <p class="small-muted">
    Predict how a recipe might be rated based on its ingredients and nutrition values.
    The app uses the Gradient Boosting champion model and actual SHAP values to show
    what pushes the predicted rating up or down.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("## Model overview")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Champion", metadata["champion_model_name"])

with m2:
    st.metric("Test MAE", f"{metadata['test_MAE']:.3f}")

with m3:
    st.metric("Test RMSE", f"{metadata['test_RMSE']:.3f}")

with m4:
    st.metric("Test R²", f"{metadata['test_R2']:.3f}")


# --------------------------------------------------
# Prediction + SHAP output
# --------------------------------------------------

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

    st.markdown(f"""
    <div class="metric-card">
        <div class="small-muted">Predicted recipe rating</div>
        <div class="big-rating">{prediction:.2f} / 5</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Ingredients used")
    st.info(", ".join(ingredients) if ingredients else "None")

    shap_df, expected_value, shap_explanation = shap_explain(input_row)

    st.markdown("## SHAP rating drivers")

    st.write(
        f"Model baseline prediction: **{expected_value:.2f}**. "
        "The SHAP plot shows which ingredients and features push the predicted rating higher or lower."
    )

    st.markdown("### SHAP waterfall plot")

    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_explanation, max_display=12, show=False)
    st.pyplot(fig)
    plt.close(fig)

    active_or_important = shap_df.head(20).copy()

    positive = active_or_important[active_or_important["shap_value"] > 0].head(10)
    negative = (
        active_or_important[active_or_important["shap_value"] < 0]
        .sort_values("shap_value")
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="positive-box">
            <h3>⬆️ Pushing rating up</h3>
            <p class="small-muted">Features with positive SHAP values.</p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            positive[["feature_clean", "shap_value"]],
            use_container_width=True
        )

    with col2:
        st.markdown("""
        <div class="negative-box">
            <h3>⬇️ Pushing rating down</h3>
            <p class="small-muted">Features with negative SHAP values.</p>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            negative[["feature_clean", "shap_value"]],
            use_container_width=True
        )

    st.markdown("### SHAP bar chart")

    plot_df = pd.concat([positive, negative]).drop_duplicates("feature")
    plot_df = plot_df.sort_values("shap_value")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["feature_clean"], plot_df["shap_value"])
    ax.axvline(0)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title("Features driving the predicted rating")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("See full SHAP table"):
        st.dataframe(
            shap_df[["feature_clean", "value", "shap_value"]],
            use_container_width=True
        )

else:
    st.info("Enter ingredients and nutrition values in the sidebar, then click Predict rating.")
