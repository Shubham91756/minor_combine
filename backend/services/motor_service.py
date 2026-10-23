from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import logging

_DEFAULT_MODEL_PATH = Path('models/trained_models/motor_baseline_full/motor_rf_further_retrained.joblib')
MODEL_PATH = Path(os.environ.get('MOTOR_MODEL_PATH') or str(_DEFAULT_MODEL_PATH))
LABELS_FILE = MODEL_PATH.parent / 'labels.json'
try:
    MOTOR_THRESHOLD = float(os.environ.get('MOTOR_THRESHOLD', '0.5'))
except Exception:
    MOTOR_THRESHOLD = 0.5


# store the last load error traceback (string) when loading fails
# declared at module top so functions can assign to it with 'global'
_LAST_LOAD_ERROR = None


def load_model():
    # Be defensive: loading a pickled sklearn model can fail if the model was
    # saved with a different scikit-learn version. Instead of letting that
    # crash application startup, catch the exception, log it, and return
    # (None, None, None). The rest of the code will raise a clearer error when
    # prediction is attempted.
    try:
        art = joblib.load(str(MODEL_PATH))
    except Exception as e:
        # capture full traceback for later inspection via the /info endpoint
        import traceback as _tb
        _err = _tb.format_exc()
        logging.getLogger(__name__).exception('Failed to load motor model %s: %s', MODEL_PATH, e)
        # store the last load error for diagnostics
        try:
            globals()['_LAST_LOAD_ERROR'] = _err
        except Exception:
            pass
        return None, None, None
    # if we reach here, load succeeded — clear any previous load error
    try:
        globals()['_LAST_LOAD_ERROR'] = None
    except Exception:
        pass
    # art could be dict or raw model
    if isinstance(art, dict) and 'model' in art:
        model = art['model']
        feats = art.get('features', None)
        # try to find a labels mapping in artifact
        labels = art.get('labels') or art.get('class_mapping') or art.get('label_map')
        # if an external labels.json exists next to the model, prefer it
        try:
            if LABELS_FILE.exists():
                import json
                with LABELS_FILE.open('r', encoding='utf8') as fh:
                    labels = json.load(fh)
                # normalize dict keys to ints when labels.json uses string keys
                if isinstance(labels, dict):
                    new = {}
                    for k, v in labels.items():
                        try:
                            ik = int(k)
                        except Exception:
                            ik = k
                        new[ik] = v
                    labels = new
        except Exception:
            pass
        return model, feats, labels
    # raw model: no features or labels saved
    return art, None, None


_MODEL = None
_FEATURES = None
_LABELS = None


def ensure_loaded():
    global _MODEL, _FEATURES
    global _LABELS
    if _MODEL is None:
        _MODEL, _FEATURES, _LABELS = load_model()
    return _MODEL, _FEATURES, _LABELS


def get_load_error():
    """Return the last exception traceback (string) captured while attempting
    to load the motor model, or None if the last load succeeded.
    """
    return _LAST_LOAD_ERROR



def _confidence_level_from_std(std):
    if std is None or not np.isfinite(std):
        return 'n/a'
    if std < 0.03:
        return 'High'
    if std < 0.08:
        return 'Medium'
    return 'Low'


def _patient_like_prob_from_proba(classes, proba_row):
    try:
        s = 0.0
        for i, c in enumerate(classes):
            try:
                ic = int(c)
            except Exception:
                continue
            if ic != 0:
                s += float(proba_row[i])
        return float(s)
    except Exception:
        return None


def _forest_prob_samples_patient_like(model, X):
    """Return per-tree patient-like probability samples when possible."""
    try:
        ests = getattr(model, 'estimators_', None)
        if not ests:
            return None
        classes = list(getattr(model, 'classes_', []))
        samples = []
        for est in ests:
            row = est.predict_proba(X)[0]
            p = _patient_like_prob_from_proba(classes, row)
            if p is not None:
                samples.append(p)
        if not samples:
            return None
        return np.asarray(samples, dtype=np.float32)
    except Exception:
        return None


def _top_features_global(model, feature_names, feature_values, k=10):
    try:
        imps = getattr(model, 'feature_importances_', None)
        if imps is None:
            return []
        imps = np.asarray(imps)
        idx = np.argsort(imps)[::-1][:k]
        out = []
        for i in idx:
            name = feature_names[i] if feature_names and i < len(feature_names) else f'feature_{i}'
            val = None
            if feature_values is not None and i < len(feature_values):
                try:
                    val = float(feature_values[i])
                except Exception:
                    val = None
            out.append({'name': name, 'importance': float(imps[i]), 'value': val})
        return out
    except Exception:
        return []

def predict_from_features_dict(features_dict):
    """Accepts a mapping of feature_name->value and returns prediction.

    Important:
    - The underlying motor model is **3-class** (healthy / patient / other).
    - For product/demo UX we also return a **binary projection** (healthy vs patient-like).

    Output convention:
    - Top-level `pred`/`label`/`prob_pos` represent the **binary** decision.
    - `multi_pred`/`multi_label` and `probs` preserve the original 3-class result.
    """
    model, feats, labels = ensure_loaded()
    if model is None:
        raise RuntimeError(
            f"Motor model not loaded. Expected model at {MODEL_PATH}. "
            "This can happen if the saved model is incompatible with the installed scikit-learn version. "
            "Either install the scikit-learn version used to create the model or retrain the model in this environment."
        )

    if not isinstance(features_dict, dict) or len(features_dict) == 0:
        raise RuntimeError(
            "Motor features are empty. Provide a feature JSON (or use the demo JSON) with numeric values."
        )

    # Create vector in the order of feats if provided, else use sorted keys.
    if feats:
        # Use a DataFrame with feature names to match training and avoid sklearn warnings
        X = pd.DataFrame([[features_dict.get(f, 0.0) for f in feats]], columns=feats)
    else:
        keys = sorted(features_dict.keys())
        X = np.array([[features_dict[k] for k in keys]], dtype=float)

    pred = model.predict(X)
    pred_idx = int(pred[0])

    # Probabilities
    try:
        proba_all = model.predict_proba(X)
    except Exception:
        proba_all = None

    probs = None
    if proba_all is not None:
        probs = {int(cls): float(prob) for cls, prob in zip(model.classes_, proba_all[0])}

    # Resolve multi-class label
    if labels:
        if isinstance(labels, dict):
            multi_label = labels.get(pred_idx, str(pred_idx))
        elif isinstance(labels, (list, tuple)):
            multi_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
        else:
            multi_label = str(pred_idx)
    else:
        multi_label = f'class_{pred_idx}'

    # Binary projection for demo/product:
    # Treat anything non-healthy (class 0) as "patient-like".
    prob_pd = probs.get(1) if probs is not None else None
    prob_other = probs.get(2) if probs is not None else None
    prob_patient = None
    if probs is not None:
        prob_patient = float(prob_pd or 0.0) + float(prob_other or 0.0)

    if prob_patient is not None:
        binary_pred = 1 if float(prob_patient) >= MOTOR_THRESHOLD else 0
    else:
        binary_pred = 0 if pred_idx == 0 else 1
    binary_label = 'healthy' if binary_pred == 0 else 'patient'

    resp = {
        # Binary (UI-facing)
        'pred': int(binary_pred),
        'label': binary_label,
        'prob_pos': prob_patient,
        'threshold': float(MOTOR_THRESHOLD),

        # Multi-class (diagnostics)
        'multi_pred': int(pred_idx),
        'multi_label': multi_label,
        'prob_pd': prob_pd,
        'prob_other': prob_other,
        'prob_patient': prob_patient,
        'probs': probs,
        'model_path': str(MODEL_PATH),
    }

    # Confidence from tree dispersion (when available)
    try:
        prob_samples = _forest_prob_samples_patient_like(model, X)
        std = float(np.std(prob_samples)) if prob_samples is not None else None
        mean = float(np.mean(prob_samples)) if prob_samples is not None else (float(prob_patient) if prob_patient is not None else None)
    except Exception:
        std = None
        mean = float(prob_patient) if prob_patient is not None else None

    confidence_interval = {
        'mean': mean,
        'std': std,
        'level': _confidence_level_from_std(std),
        'method': 'rf_tree_dispersion' if std is not None else 'single_probability',
    }

    # Global feature importance (model-level) + current feature values
    try:
        feature_names = feats if feats else None
        if feats:
            feature_values = [features_dict.get(f, 0.0) for f in feats]
        else:
            feature_values = None
        top_features = _top_features_global(model, feature_names, feature_values, k=10)
    except Exception:
        top_features = []

    resp['confidence_interval'] = confidence_interval
    resp['reasoning'] = {
        'top_features': top_features,
        'heatmap_url': None,
        'confidence_interval': confidence_interval,
        'notes': 'Top features are global importances from the motor model; values are from the uploaded feature vector.',
    }

    try:
        import os
        from datetime import datetime
        resp['server_pid'] = os.getpid()
        resp['server_time'] = datetime.utcnow().isoformat()
    except Exception:
        pass

    return resp


def predict_from_csv(file_path):
    df = pd.read_csv(file_path)
    model, feats, labels = ensure_loaded()
    if model is None:
        raise RuntimeError(
            f"Motor model not loaded. Expected model at {MODEL_PATH}. "
            "This can happen if the saved model is incompatible with the installed scikit-learn version. "
            "Either install the scikit-learn version used to create the model or retrain the model in this environment."
        )

    # Build feature matrix X in the expected feature order.
    if feats is not None:
        df_sub = df.reindex(columns=feats)
        for c in df_sub.columns:
            try:
                df_sub[c] = pd.to_numeric(df_sub[c], errors='coerce')
            except Exception:
                df_sub[c] = np.nan
        X = df_sub.fillna(0.0).values
    else:
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            coerced = df.copy()
            for c in coerced.columns:
                coerced[c] = pd.to_numeric(coerced[c], errors='coerce')
            numeric_df = coerced.select_dtypes(include=[np.number])
        if numeric_df.shape[1] == 0:
            raise RuntimeError('No numeric features found in uploaded CSV for motor prediction')
        X = numeric_df.fillna(0.0).values

    preds = model.predict(X)
    try:
        prob_all = model.predict_proba(X)
    except Exception:
        prob_all = None

    # Helper for label mapping
    def resolve_label(i: int):
        if labels:
            if isinstance(labels, dict):
                return labels.get(i, str(i))
            if isinstance(labels, (list, tuple)):
                return labels[i] if i < len(labels) else str(i)
        return f'class_{i}'

    out = pd.DataFrame({'multi_pred': [int(p) for p in preds]})
    out['multi_label'] = [resolve_label(int(p)) for p in preds]

    if prob_all is not None:
        classes = [int(c) for c in model.classes_]
        # per-row class prob dict
        out['probs'] = [{int(c): float(prob) for c, prob in zip(classes, row)} for row in prob_all]
        # pd/other/abnormal probabilities
        def get_prob(row, cls):
            try:
                idx = classes.index(cls)
                return float(row[idx])
            except Exception:
                return 0.0
        out['prob_pd'] = [get_prob(row, 1) for row in prob_all]
        out['prob_other'] = [get_prob(row, 2) for row in prob_all]
        out['prob_patient'] = out['prob_pd'] + out['prob_other']
    else:
        out['probs'] = [None] * len(out)
        out['prob_pd'] = [None] * len(out)
        out['prob_other'] = [None] * len(out)
        out['prob_patient'] = [None] * len(out)

    # Binary projection
    if prob_all is not None:
        thr = float(MOTOR_THRESHOLD)
        out['binary_pred'] = [1 if (p is not None and float(p) >= thr) else 0 for p in out['prob_patient']]
    else:
        out['binary_pred'] = [0 if int(p) == 0 else 1 for p in preds]
    out['binary_label'] = ['healthy' if bp == 0 else 'patient' for bp in out['binary_pred']]

    # Backward-compatible UI fields
    out['pred'] = out['binary_pred']
    out['label'] = out['binary_label']
    out['prob_pos'] = out['prob_patient']
    out['threshold'] = float(MOTOR_THRESHOLD)

    # Optional reasoning-lite (global importances only)
    try:
        model0 = model
        feature_names = feats if feats is not None else None
        top_features = _top_features_global(model0, feature_names, None, k=10)
    except Exception:
        top_features = []
    out['reasoning'] = [{'top_features': top_features, 'heatmap_url': None, 'confidence_interval': None} for _ in range(len(out))]

    return out


# ------------------------------------------------------------------
# Motor timeseries (single-file) prediction
# ------------------------------------------------------------------

def _motor_file_features(file_path: str):
    """Extract the same per-file features used in training.

    This mirrors `models/trainers/train_motor_reextract.py:file_features` so
    a single timeseries file can be classified without precomputing a full
    per-subject aggregation.
    """
    import pandas as pd

    # read with comma sep; fallback to whitespace
    try:
        df = pd.read_csv(file_path, sep=",", header=None)
    except Exception:
        df = pd.read_csv(file_path, sep=r"\s+", header=None)

    # if only one column read, try to split by comma manually
    if df.shape[1] == 1:
        try:
            df = df[0].astype(str).str.split(",", expand=True).astype(float)
        except Exception:
            # last resort: keep as-is
            pass

    df = df.apply(pd.to_numeric, errors="coerce")

    out = {}
    ncols = df.shape[1]
    sig_cols = list(range(1, min(7, ncols)))  # 1..6 if present

    for i in sig_cols:
        col = df.iloc[:, i].dropna()
        prefix = f"c{i}"
        out[f"{prefix}_mean"] = float(col.mean()) if not col.empty else 0.0
        out[f"{prefix}_std"] = float(col.std()) if not col.empty else 0.0
        out[f"{prefix}_median"] = float(col.median()) if not col.empty else 0.0
        out[f"{prefix}_min"] = float(col.min()) if not col.empty else 0.0
        out[f"{prefix}_max"] = float(col.max()) if not col.empty else 0.0
        out[f"{prefix}_ptp"] = float(col.max() - col.min()) if not col.empty else 0.0
        out[f"{prefix}_rms"] = float(np.sqrt((col ** 2).mean())) if not col.empty else 0.0
        out[f"{prefix}_energy"] = float((col ** 2).sum()) if not col.empty else 0.0

    # magnitude features if at least 3 channels (acc 1-3)
    if len(sig_cols) >= 3:
        acc = df.iloc[:, 1:4].astype(float).fillna(0).values
        mag = np.linalg.norm(acc, axis=1)
        out["acc_mag_mean"] = float(mag.mean())
        out["acc_mag_std"] = float(mag.std())
    else:
        out["acc_mag_mean"] = 0.0
        out["acc_mag_std"] = 0.0

    t = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    out["nrows"] = int(df.shape[0])
    out["duration"] = float(t.max() - t.min()) if t.notnull().any() else 0.0
    dts = t.diff().dropna()
    out["median_dt"] = float(dts.median()) if not dts.empty else 0.0

    return out


def predict_from_timeseries_file(file_path: str):
    """Predict from a single raw motor timeseries `.txt` file.

    Returns the same response schema as `predict_from_features_dict` (binary + multi-class).
    """
    feats = _motor_file_features(file_path)
    return predict_from_features_dict(feats)

