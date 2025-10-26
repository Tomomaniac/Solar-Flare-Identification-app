
# pages/Advanced_Prediction.py
import streamlit as st
from pathlib import Path
import joblib
import pickle
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import re
import os

# Try to reuse your existing helpers from flare_detector_app.py
try:
    from flare_detector_app import preprocess_image, apply_colormap, extract_features_v1, extract_features_v3
    _HAS_MAIN_HELPERS = True
except Exception:
    _HAS_MAIN_HELPERS = False

# Fallback helpers (only used if your main app helpers are not importable)
if not _HAS_MAIN_HELPERS:
    def preprocess_image(img_array):
        img_min = np.min(img_array)
        img_max = np.max(img_array)
        if img_max - img_min == 0:
            return np.zeros_like(img_array, dtype=float)
        return (img_array - img_min) / (img_max - img_min)

    def apply_colormap(img_normalized, colormap='hot', cm=cm):
        cmap = cm.get_cmap(colormap)
        img_colored = cmap(img_normalized)
        return (img_colored[:, :, :3] * 255).astype(np.uint8)

    def extract_features_v1(img_normalized):
        feats = {}
        feats['max_brightness'] = float(np.max(img_normalized))
        feats['mean_brightness'] = float(np.mean(img_normalized))
        feats['std_brightness'] = float(np.std(img_normalized))
        thr = 0.9 * feats['max_brightness']
        coords = np.argwhere(img_normalized >= thr)
        if len(coords) > 0:
            feats['centroid_y'] = float(np.mean(coords[:, 0]))
            feats['centroid_x'] = float(np.mean(coords[:, 1]))
        else:
            H, W = img_normalized.shape
            feats['centroid_y'] = float(H/2)
            feats['centroid_x'] = float(W/2)
        return feats

    def extract_features_v3(img_normalized, img_raw):
        feats = extract_features_v1(img_normalized)
        feats['raw_max'] = float(np.max(img_raw))
        feats['raw_mean'] = float(np.mean(img_raw))
        return feats

# Configuration
MODELS_DIR = Path("models")
FEATURE_PKL = MODELS_DIR / "feature_columns_prediction.pkl"
VALID_FILTERS = ("131", "193")  # filters we expect

st.set_page_config(page_title="Advanced Prediction Methods", layout="centered")
st.title("🧪 Advanced Prediction Methods (model groups)")

# Discover models and group them by base name
def discover_model_groups(models_dir=MODELS_DIR):
    groups = {}  # base_name -> { '131': path, '193': path }
    if not models_dir.exists():
        return groups
    for p in models_dir.glob("*.pkl"):
        m = re.match(r"(.+?)_(131|193)\.pkl$", p.name, flags=re.IGNORECASE)
        if m:
            base = m.group(1)
            filt = m.group(2)
            base = base.strip()
            groups.setdefault(base, {})[filt] = p
    return groups

model_groups = discover_model_groups()

if not model_groups:
    st.error("No model files found in models/ with pattern <name>_131.pkl or <name>_193.pkl. Place your models in models/ using that pattern.")
    st.stop()

# Load feature columns ordering
if not FEATURE_PKL.exists():
    st.error(f"Feature column file not found: {FEATURE_PKL}\nThis page uses models/feature_columns_prediction.pkl for feature ordering.")
    st.stop()

with open(FEATURE_PKL, "rb") as f:
    feature_cols_global = pickle.load(f)

# Sidebar or top UI selectors
group_name = st.selectbox("Select model group (algorithm)", sorted(model_groups.keys()))
available_filters = sorted(model_groups[group_name].keys())
filter_choice = st.selectbox("Select filter", available_filters)
colormap = st.selectbox("Colormap", ["hot", "inferno", "plasma", "magma", "viridis", "gray"])

uploaded_file = st.file_uploader("Upload Solar Image", type=['tif', 'tiff', 'png', 'jpg', 'jpeg'])
if uploaded_file is None:
    st.info("Upload an image to analyze.")
    st.stop()

# Load and show image
img = Image.open(uploaded_file)
img_array = np.array(img)
img_normalized = preprocess_image(img_array)
img_display = apply_colormap(img_normalized, colormap, cm=cm)
st.image(img_display, caption=uploaded_file.name, use_container_width=True)

# Helper: load model (supports joblib/pickle and packaged dicts)
def load_model_package(path: Path):
    # returns tuple (model_object, scaler_or_None, feature_cols_or_None)
    try:
        obj = joblib.load(str(path))
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f)

    # If it's a packaged dict with keys 'model', 'scaler', 'feature_cols'
    if isinstance(obj, dict):
        model = obj.get('model') or obj.get('estimator') or obj.get('clf') or obj.get('classifier')
        scaler = obj.get('scaler')
        fc = obj.get('feature_cols') or obj.get('feature_columns')
        # if the top-level object itself is a sklearn model, return it
        if model is None and hasattr(obj, "predict"):
            model = obj
        return model, scaler, fc
    else:
        # obj is likely the raw model
        return obj, None, None

# Build feature vector using feature_columns_prediction.pkl by default, fallback to package feature_cols if present
def build_feature_vector(img_norm, img_raw, feature_cols_list):
    # produce merged feature dict
    fv1 = extract_features_v1(img_norm)
    fv3 = extract_features_v3(img_norm, img_raw)
    feats = {**fv1, **fv3}  # v3 overrides v1 if duplicates
    X = []
    missing_features = []
    for col in feature_cols_list:
        if col in feats:
            X.append(feats[col])
        else:
            X.append(0.0)
            missing_features.append(col)
    X_arr = np.array(X, dtype=float).reshape(1, -1)
    return X_arr, missing_features, feats

# Load selected model
model_path = model_groups[group_name][filter_choice]
model, scaler, model_feature_cols = load_model_package(model_path)

# Choose which feature order to use
if model_feature_cols:
    feature_cols_used = model_feature_cols
else:
    feature_cols_used = feature_cols_global

st.markdown(f"**Using feature columns file:** {FEATURE_PKL.name}  " if not model_feature_cols else f"**Using packaged feature columns from model**")

if st.button("Analyze with selected model"):
    with st.spinner("Computing features and running model..."):
        X, missing, feats_dict = build_feature_vector(img_normalized, img_array, feature_cols_used)

        if missing:
            st.warning(f"{len(missing)} features from feature list were not found and defaulted to 0. Examples: {missing[:8]}")

        # Apply scaler if present
        if scaler is not None:
            try:
                X = scaler.transform(X)
            except Exception as e:
                st.warning(f"Failed to apply scaler from package: {e}")

        # Run prediction
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            st.stop()

        probs = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
        except Exception:
            probs = None

        # Output
        st.subheader("Results")
        st.write(f"Model group: **{group_name}**  •  Filter: **{filter_choice} Å**")
        st.write(f"Prediction: **{pred}**")
        if probs is not None:
            probs_list = [float(round(p, 4)) for p in probs.tolist()]
            st.write("Probabilities:", probs_list)
            st.metric("Top probability", f"{max(probs_list):.4f}")

        # show some computed features
        with st.expander("Show computed features (sample)"):
            sample_show = {k: feats_dict[k] for k in list(feats_dict.keys())[:20]}
            st.json(sample_show)

        st.success("Done")
