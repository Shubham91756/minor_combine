import io
from pathlib import Path
import json
import numpy as np
from PIL import Image

try:
    from tensorflow import keras
    from tensorflow.keras import layers
except ModuleNotFoundError:
    keras = None
    layers = None


# ------------------------------------------------------------------
# Ensemble model configuration
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / 'models' / 'handwriting_ensemble'
MODEL_A_PATH = MODEL_DIR / 'parkinson_model_best_resnet50.keras'
MODEL_B_PATH = MODEL_DIR / 'parkinson_model_best_efficientnetb2.keras'
THRESHOLD_PATH = MODEL_DIR / 'best_threshold_ensemble.json'

DEFAULT_IMG_SIZE = 224
DEFAULT_THRESHOLD = 0.5
DEFAULT_WEIGHT_RESNET = 0.5

CLASS_NAMES = ["healthy", "patient"]


# ------------------------------------------------------------------
# Custom preprocessing layer (matches training)
# ------------------------------------------------------------------
if keras is not None:
    @keras.utils.register_keras_serializable()
    class PreprocessInput(layers.Layer):
        def __init__(self, model_name, **kwargs):
            super().__init__(**kwargs)
            self.model_name = model_name

        def call(self, inputs):
            x = inputs * 255.0
            if self.model_name == "ResNet50":
                from tensorflow.keras.applications.resnet50 import preprocess_input
                return preprocess_input(x)
            if self.model_name == "MobileNetV2":
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
                return preprocess_input(x)
            if self.model_name == "DenseNet121":
                from tensorflow.keras.applications.densenet import preprocess_input
                return preprocess_input(x)
            if self.model_name == "EfficientNetB2":
                from tensorflow.keras.applications.efficientnet import preprocess_input
                return preprocess_input(x)
            return x

        def get_config(self):
            config = super().get_config()
            config.update({"model_name": self.model_name})
            return config
else:
    PreprocessInput = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_threshold_and_weight():
    threshold = DEFAULT_THRESHOLD
    weight_resnet = DEFAULT_WEIGHT_RESNET
    if THRESHOLD_PATH.exists():
        try:
            data = json.loads(THRESHOLD_PATH.read_text())
            threshold = float(data.get("threshold", threshold))
            weight_resnet = float(data.get("weight_resnet50", weight_resnet))
        except Exception:
            pass
    return threshold, weight_resnet


def image_to_model_input(img: Image.Image, img_size: int):
    img = img.convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)


def _confidence_level_from_std(std: float | None):
    if std is None or not np.isfinite(std):
        return 'n/a'
    if std < 0.03:
        return 'High'
    if std < 0.08:
        return 'Medium'
    return 'Low'


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------
_MODEL_A = None
_MODEL_B = None
_THRESHOLD = None
_WEIGHT_RESNET = None


def load_models():
    if keras is None:
        raise RuntimeError("TensorFlow is not installed in this environment. Install it with: pip install tensorflow")

    if not MODEL_A_PATH.exists() or not MODEL_B_PATH.exists():
        raise RuntimeError(
            f"Ensemble model files not found. Expected:\n{MODEL_A_PATH}\n{MODEL_B_PATH}"
        )

    model_a = keras.models.load_model(
        str(MODEL_A_PATH),
        custom_objects={"PreprocessInput": PreprocessInput},
        compile=False,
    )
    model_b = keras.models.load_model(
        str(MODEL_B_PATH),
        custom_objects={"PreprocessInput": PreprocessInput},
        compile=False,
    )

    threshold, weight_resnet = load_threshold_and_weight()
    return model_a, model_b, threshold, weight_resnet


def ensure_loaded():
    global _MODEL_A, _MODEL_B, _THRESHOLD, _WEIGHT_RESNET
    if _MODEL_A is None or _MODEL_B is None:
        _MODEL_A, _MODEL_B, _THRESHOLD, _WEIGHT_RESNET = load_models()
    return _MODEL_A, _MODEL_B, _THRESHOLD, _WEIGHT_RESNET


# ------------------------------------------------------------------
# Grad-CAM (best-effort)
# ------------------------------------------------------------------

def _find_last_conv_layer_name(model):
    if keras is None:
        return None
    try:
        for layer in reversed(model.layers):
            try:
                if isinstance(layer, keras.layers.Conv2D):
                    return layer.name
            except Exception:
                continue
        return None
    except Exception:
        return None


def _gradcam_heatmap(model, X, layer_name=None):
    """Compute a Grad-CAM heatmap for a binary sigmoid output model.

    Returns a float32 HxW array in [0,1].
    """
    if keras is None:
        return None
    import tensorflow as tf

    try:
        if layer_name is None:
            layer_name = _find_last_conv_layer_name(model)
        if layer_name is None:
            return None

        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

        X_tf = tf.convert_to_tensor(X)
        with tf.GradientTape() as tape:
            conv_out, pred = grad_model(X_tf)
            # pred shape (1,1)
            loss = pred[:, 0]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        denom = tf.reduce_max(heatmap)
        heatmap = heatmap / (denom + 1e-8)
        return heatmap.numpy().astype(np.float32)
    except Exception:
        return None


def _heatmap_png_rgba(heatmap01: np.ndarray, out_w: int, out_h: int):
    """Create an RGBA PNG overlay from a heatmap in [0,1]."""
    try:
        hm = np.nan_to_num(heatmap01, nan=0.0)
        hm = np.clip(hm, 0.0, 1.0)
        # upscale to requested size
        hm_img = Image.fromarray((hm * 255).astype(np.uint8), mode='L').resize((out_w, out_h), resample=Image.BILINEAR)
        hm_arr = np.asarray(hm_img).astype(np.float32) / 255.0

        # simple red-yellow colormap
        r = (255 * np.ones_like(hm_arr)).astype(np.uint8)
        g = (255 * hm_arr).astype(np.uint8)
        b = (40 * (1.0 - hm_arr)).astype(np.uint8)
        a = (180 * hm_arr).astype(np.uint8)

        rgba = np.stack([r, g, b, a], axis=-1)
        out = Image.fromarray(rgba, mode='RGBA')
        buf = io.BytesIO()
        out.save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return None


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------

def predict_image(file_bytes):
    model_a, model_b, threshold, weight_resnet = ensure_loaded()

    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    X = image_to_model_input(img, DEFAULT_IMG_SIZE)

    p_a = float(model_a.predict(X, verbose=0)[0][0])
    p_b = float(model_b.predict(X, verbose=0)[0][0])
    prob_pos = (weight_resnet * p_a) + ((1 - weight_resnet) * p_b)

    pred = 1 if prob_pos >= threshold else 0
    label = CLASS_NAMES[pred]

    return {
        "label": label,
        "pred": int(pred),
        "prob_pos": float(prob_pos),
        "prob_a": float(p_a),
        "prob_b": float(p_b),
        "threshold": float(threshold),
        "weight_resnet": float(weight_resnet),
    }


def predict_image_xai(file_bytes):
    """Predict + generate Grad-CAM overlay (best-effort).

    Returns: (result_dict, heatmap_png_bytes_or_None)
    """
    model_a, model_b, threshold, weight_resnet = ensure_loaded()

    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    orig_w, orig_h = img.size

    X = image_to_model_input(img, DEFAULT_IMG_SIZE)

    p_a = float(model_a.predict(X, verbose=0)[0][0])
    p_b = float(model_b.predict(X, verbose=0)[0][0])
    prob_pos = (weight_resnet * p_a) + ((1 - weight_resnet) * p_b)

    pred = 1 if prob_pos >= threshold else 0
    label = CLASS_NAMES[pred]

    # Confidence: dispersion across ensemble members
    std = float(np.std(np.asarray([p_a, p_b], dtype=np.float32)))
    confidence_interval = {
        'mean': float(prob_pos),
        'std': std,
        'level': _confidence_level_from_std(std),
        'method': 'ensemble_member_dispersion',
    }

    heatmap_png = None
    heatmap = _gradcam_heatmap(model_a, X)
    if heatmap is not None:
        heatmap_png = _heatmap_png_rgba(heatmap, orig_w, orig_h)

    res = {
        "label": label,
        "pred": int(pred),
        "prob_pos": float(prob_pos),
        "prob_a": float(p_a),
        "prob_b": float(p_b),
        "threshold": float(threshold),
        "weight_resnet": float(weight_resnet),
        "confidence_interval": confidence_interval,
        "reasoning": {
            "top_features": [],
            "heatmap_url": None,
            "confidence_interval": confidence_interval,
            "notes": "Grad-CAM is best-effort and highlights image regions most influential to the prediction.",
        },
    }

    return res, heatmap_png
