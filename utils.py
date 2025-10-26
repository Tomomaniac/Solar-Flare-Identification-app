# utils.py
import joblib
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# --- Preprocessing and visualization helpers ---
def preprocess_image(img_array):
    """Normalize image to [0, 1] range"""
    img_min = np.min(img_array)
    img_max = np.max(img_array)
    if img_max - img_min == 0:
        return np.zeros_like(img_array, dtype=float)
    return (img_array - img_min) / (img_max - img_min)

def apply_colormap(img_normalized, colormap='hot', cm=None):
    """Apply colormap to normalized image (hot spots show red). Pass matplotlib.cm as cm."""
    cmap = cm.get_cmap(colormap) if cm is not None else None
    if cmap is None:
        raise ValueError("Provide matplotlib.cm as `cm` argument")
    img_colored = cmap(img_normalized)
    return (img_colored[:, :, :3] * 255).astype(np.uint8)

# --- Feature extractors (copy of your functions, kept small here) ---
def extract_features_v1(img_normalized):
    features = {}
    features['max_brightness'] = float(np.max(img_normalized))
    features['mean_brightness'] = float(np.mean(img_normalized))
    features['std_brightness'] = float(np.std(img_normalized))
    threshold_90 = 0.9 * features['max_brightness']
    bright_pixels_90 = img_normalized >= threshold_90
    bright_coords = np.argwhere(bright_pixels_90)
    if len(bright_coords) > 0:
        centroid_y = float(np.mean(bright_coords[:, 0]))
        centroid_x = float(np.mean(bright_coords[:, 1]))
        features['centroid_y'] = centroid_y
        features['centroid_x'] = centroid_x
        distances_from_centroid = np.sqrt(
            (bright_coords[:, 0] - centroid_y)**2 + (bright_coords[:, 1] - centroid_x)**2
        )
        features['max_distance_from_centroid'] = float(np.max(distances_from_centroid))
        features['mean_distance_from_centroid'] = float(np.mean(distances_from_centroid))
        features['std_distance_from_centroid'] = float(np.std(distances_from_centroid))
        brightest_idx = np.unravel_index(np.argmax(img_normalized), img_normalized.shape)
        distances_from_brightest = np.sqrt(
            (bright_coords[:, 0] - brightest_idx[0])**2 + (bright_coords[:, 1] - brightest_idx[1])**2
        )
        features['max_distance_from_brightest'] = float(np.max(distances_from_brightest))
        features['bright_pixels_std_y'] = float(np.std(bright_coords[:, 0]))
        features['bright_pixels_std_x'] = float(np.std(bright_coords[:, 1]))
        features['spatial_spread'] = float(np.sqrt(features['bright_pixels_std_y']**2 + features['bright_pixels_std_x']**2))
        area = len(bright_coords)
        perimeter_approx = 2 * np.pi * features['spatial_spread']
        features['compactness'] = float((perimeter_approx**2) / (4 * np.pi * area)) if area > 0 else 0.0
        y_range = np.ptp(bright_coords[:, 0])
        x_range = np.ptp(bright_coords[:, 1])
        features['aspect_ratio'] = float(max(y_range, x_range) / (min(y_range, x_range) + 1e-6))
        for pct in [90, 80, 70]:
            threshold = (pct / 100) * features['max_brightness']
            bright_pixels = img_normalized >= threshold
            features[f'concentration_{pct}'] = float(np.sum(bright_pixels) / (img_normalized.shape[0] * img_normalized.shape[1]))
        features['num_bright_pixels_90'] = int(len(bright_coords))
    else:
        # default values
        H, W = img_normalized.shape
        features.update({
            'centroid_y': float(H / 2),
            'centroid_x': float(W / 2),
            'max_distance_from_centroid': 0.0,
            'mean_distance_from_centroid': 0.0,
            'std_distance_from_centroid': 0.0,
            'max_distance_from_brightest': 0.0,
            'bright_pixels_std_y': 0.0,
            'bright_pixels_std_x': 0.0,
            'spatial_spread': 0.0,
            'compactness': 0.0,
            'aspect_ratio': 1.0,
            'concentration_90': 0.0,
            'concentration_80': 0.0,
            'concentration_70': 0.0,
            'num_bright_pixels_90': 0
        })
    return features

def extract_features_v3(img_normalized, img_raw):
    features = extract_features_v1(img_normalized)
    features['raw_max'] = float(np.max(img_raw))
    features['raw_mean'] = float(np.mean(img_raw))
    return features

# --- Model loader ---
def load_models(models_dir="models"):
    """
    Load models and feature column lists from models_dir.
    Expects model files to follow naming convention:
      - model_v1_131.pkl, model_v3_193.pkl, etc.
      - For new algorithms use names like: algoA_131.pkl, algoA_193.pkl, algoB_131.pkl, ...
    Returns: dict of models, and dict of feature columns per model family.
    """
    models_dir = Path(models_dir)
    models = {}
    feature_cols = {}

    # existing sets (v1, v3)
    mapping = {
        'v1_131': "models/model_v1_131.pkl",
        'v1_193': "models/model_v1_193.pkl",
        'v3_131': "models/model_v3_131.pkl",
        'v3_193': "models/model_v3_193.pkl",
    }

    # Add new algorithm names here if you saved them (example: algo1..algo4)
    # The user should save:
    # models/algo1_131.pkl, models/algo1_193.pkl, ..., models/algo4_193.pkl
    new_algos = ['algo1', 'algo2', 'algo3', 'algo4']  # change names as you prefer
    for algo in new_algos:
        mapping[f"{algo}_131"] = f"models/{algo}_131.pkl"
        mapping[f"{algo}_193"] = f"models/{algo}_193.pkl"

    # load models if present
    for key, path in mapping.items():
        p = Path(path)
        if p.exists():
            try:
                models[key] = joblib.load(str(p))
            except Exception:
                # try pickle if joblib fails
                with open(p, 'rb') as f:
                    models[key] = pickle.load(f)
        else:
            # skip missing models (log or keep None)
            models[key] = None

    # load feature columns files if present (common names)
    if (models_dir / "feature_columns_v1.pkl").exists():
        with open(models_dir / "feature_columns_v1.pkl", 'rb') as f:
            feature_cols['v1'] = pickle.load(f)
    if (models_dir / "feature_columns_v3.pkl").exists():
        with open(models_dir / "feature_columns_v3.pkl", 'rb') as f:
            feature_cols['v3'] = pickle.load(f)

    # for new algos, you can keep a shared feature column file (e.g., feature_columns_algo1.pkl)
    for algo in new_algos:
        fc_path = models_dir / f"feature_columns_{algo}.pkl"
        if fc_path.exists():
            with open(fc_path, 'rb') as f:
                feature_cols[algo] = pickle.load(f)

    return models, feature_cols
