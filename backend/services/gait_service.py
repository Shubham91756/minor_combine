from pathlib import Path
import os
import joblib
import numpy as np
import sklearn
from models.utils.dataset import load_gait_file, extract_gait_features_enhanced

# Allow overriding the gait model path via environment variable for easier testing/deployment
_DEFAULT_MODEL = Path('models/trained_models/gait_signals/model_gait_enhanced_calibrated.joblib')
MODEL_PATH = Path(os.environ.get('GAIT_MODEL_PATH') or str(_DEFAULT_MODEL))

# Decision threshold used by the UI (kept fixed unless explicitly configured)
try:
    GAIT_THRESHOLD = float(os.environ.get('GAIT_THRESHOLD', '0.5'))
except Exception:
    GAIT_THRESHOLD = 0.5

_MODEL = None


def ensure_loaded():
    """Load the gait model (cached).

    On error, raise a clear RuntimeError with context about sklearn versions.
    """
    global _MODEL
    if _MODEL is None:
        try:
            _MODEL = joblib.load(str(MODEL_PATH))
        except Exception as e:
            try:
                running = sklearn.__version__
            except Exception:
                running = 'unknown'
            msg = (
                f"Failed to load gait model at {MODEL_PATH!s}: {e!s}. "
                f"Running scikit-learn: {running}. If the model was trained with a different scikit-learn version, "
                "please install the matching version or retrain the model."
            )
            raise RuntimeError(msg) from e
    return _MODEL


def _confidence_level_from_std(std):
    if std is None or not np.isfinite(std):
        return 'n/a'
    # Heuristic: lower dispersion across ensemble members/trees -> higher confidence.
    if std < 0.03:
        return 'High'
    if std < 0.08:
        return 'Medium'
    return 'Low'


def _forest_prob_samples(model, X):
    """Return per-estimator probability samples when the model is a RandomForest-like estimator."""
    try:
        ests = getattr(model, 'estimators_', None)
        if not ests:
            return None
        samples = []
        for est in ests:
            p = est.predict_proba(X)[0]
            # binary case: class-1 prob is index 1
            if len(p) >= 2:
                samples.append(float(p[1]))
        if not samples:
            return None
        return np.asarray(samples, dtype=np.float32)
    except Exception:
        return None


def _gait_feature_names(n_cols: int):
    # base features = [means, stds, maxs, mins] for each column
    names = []
    for stat in ('mean', 'std', 'max', 'min'):
        for i in range(n_cols):
            if i == 0:
                names.append(f'time_{stat}')
            else:
                names.append(f'col{i}_{stat}')

    # COP stats for l_x,l_y,r_x,r_y each: mean,std,min,max
    for side in ('l', 'r'):
        for axis in ('x', 'y'):
            for stat in ('mean', 'std', 'min', 'max'):
                names.append(f'cop_{side}_{axis}_{stat}')

    # Steps + cadence + spectral energy
    names.extend(['step_count', 'mean_step_interval_s', 'cadence_spm', 'spectral_energy'])
    return names


def _top_features(model, feats_vec, feature_names, k=10):
    try:
        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            return []
        importances = np.asarray(importances)
        idx = np.argsort(importances)[::-1][:k]
        out = []
        for i in idx:
            name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            val = float(feats_vec[i]) if i < len(feats_vec) else None
            out.append({'name': name, 'importance': float(importances[i]), 'value': val})
        return out
    except Exception:
        return []


def predict_from_timeseries_file(file_path):
    """Predict gait risk from a gait timeseries file.

    Response is backward-compatible with the existing UI and includes a `reasoning` object
    for clinical transparency.
    """
    model = ensure_loaded()

    arr = load_gait_file(file_path)
    feats = extract_gait_features_enhanced(arr)
    X = feats.reshape(1, -1)

    pred = int(model.predict(X)[0])
    try:
        p = float(model.predict_proba(X)[0][1])
    except Exception:
        p = None

    # Confidence via dispersion across trees (when available)
    prob_samples = _forest_prob_samples(model, X)
    std = float(np.std(prob_samples)) if prob_samples is not None else None
    mean = float(np.mean(prob_samples)) if prob_samples is not None else (float(p) if p is not None else None)

    feature_names = _gait_feature_names(int(arr.shape[1]))
    top_features = _top_features(model, feats, feature_names, k=10)

    out = {
        'pred': int(pred),
        'label': 'patient' if pred == 1 else 'healthy',
        'prob_pos': (float(p) if p is not None else None),
        'threshold': float(GAIT_THRESHOLD),
        'model_path': str(MODEL_PATH),
        'sklearn_version': getattr(sklearn, '__version__', None),
    }

    # human-friendly verdict (kept for UI)
    try:
        if out['prob_pos'] is None:
            out['verdict'] = 'Unknown'
        elif out['prob_pos'] > 0.6:
            out['verdict'] = 'Likely patient'
        elif out['prob_pos'] < 0.4:
            out['verdict'] = 'Likely healthy'
        else:
            out['verdict'] = 'Borderline — review'
    except Exception:
        out['verdict'] = 'Unknown'

    confidence_interval = {
        'mean': mean,
        'std': std,
        'level': _confidence_level_from_std(std),
        'method': 'rf_tree_dispersion' if prob_samples is not None else 'single_probability',
    }

    out['confidence_interval'] = confidence_interval
    out['reasoning'] = {
        'top_features': top_features,
        'heatmap_url': None,
        'confidence_interval': confidence_interval,
        'notes': 'Top features are global importances from the gait model; values are from the uploaded signal after feature extraction.',
    }

    return out
