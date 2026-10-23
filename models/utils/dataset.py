import os
import numpy as np
from PIL import Image
import json
import pandas as pd

def load_gait_file(path):
    # TSV with numeric columns, first column is time
    data = pd.read_csv(path, sep='\t', header=None)
    return data.values

def extract_gait_features(arr):
    # arr: NxM numeric array
    # simple features: mean, std, skew (via scipy if available), max, min for each column
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    mins = np.nanmin(arr, axis=0)
    feats = np.concatenate([means, stds, maxs, mins])
    return feats


# --- Enhanced gait utilities ---
# sensor coordinates from format.txt (arbitrary scale used by dataset)
L_X = np.array([-500, -700, -300, -700, -300, -700, -300, -500], dtype=np.float32)
L_Y = np.array([-800, -400, -400,    0,    0,  400,  400,  800], dtype=np.float32)
R_X = np.array([ 500,  700,  300,  700,  300,  700,  300,  500], dtype=np.float32)
R_Y = np.array([-800, -400, -400,    0,    0,  400,  400,  800], dtype=np.float32)

def compute_cop_per_sample(arr):
    """Compute center-of-pressure (COP) x,y per sample for left and right foot.

    arr is NxM where columns are: [time, L1..L8, R1..R8, total_left, total_right]
    returns: dict with keys 'l_x','l_y','r_x','r_y' each an array length N with NaNs when no force
    """
    # expected indices
    # time idx 0; left sensors 1..8 -> indices 1..8; right sensors 9..16; totals 17,18
    left_idx = slice(1,9)
    right_idx = slice(9,17)

    l_forces = np.asarray(arr[:, left_idx], dtype=np.float32)
    r_forces = np.asarray(arr[:, right_idx], dtype=np.float32)

    l_sum = np.nansum(l_forces, axis=1)
    r_sum = np.nansum(r_forces, axis=1)

    l_x = np.full(len(arr), np.nan, dtype=np.float32)
    l_y = np.full(len(arr), np.nan, dtype=np.float32)
    r_x = np.full(len(arr), np.nan, dtype=np.float32)
    r_y = np.full(len(arr), np.nan, dtype=np.float32)

    # avoid division by zero
    nonzero = l_sum > 1e-6
    if np.any(nonzero):
        l_x[nonzero] = (l_forces[nonzero] * L_X).sum(axis=1) / l_sum[nonzero]
        l_y[nonzero] = (l_forces[nonzero] * L_Y).sum(axis=1) / l_sum[nonzero]

    nonzero = r_sum > 1e-6
    if np.any(nonzero):
        r_x[nonzero] = (r_forces[nonzero] * R_X).sum(axis=1) / r_sum[nonzero]
        r_y[nonzero] = (r_forces[nonzero] * R_Y).sum(axis=1) / r_sum[nonzero]

    return {'l_x': l_x, 'l_y': l_y, 'r_x': r_x, 'r_y': r_y}


def detect_steps_from_total_force(arr, sr=100):
    """Simple step detection based on combined total force signal.

    Returns number of detected steps and mean step interval in seconds.
    This is a lightweight heuristic (local maxima above mean+0.5*std).
    """
    try:
        total = np.nansum(arr[:, 1:17], axis=1)
    except Exception:
        # fallback: if totals are provided in columns 17/18
        if arr.shape[1] >= 19:
            total = np.nan_to_num(arr[:, 17]) + np.nan_to_num(arr[:, 18])
        else:
            total = np.nansum(arr, axis=1)

    if len(total) < 3:
        return 0, np.nan

    mean = np.nanmean(total)
    std = np.nanstd(total)
    thresh = mean + 0.5 * std

    # simple local maxima detection
    peaks = []
    for i in range(1, len(total)-1):
        if total[i] > thresh and total[i] > total[i-1] and total[i] > total[i+1]:
            peaks.append(i)

    if len(peaks) < 2:
        return len(peaks), np.nan

    intervals = np.diff(peaks) / float(sr)
    return len(peaks), float(np.nanmean(intervals))


def extract_gait_features_enhanced(arr, sr=100):
    """Enhanced feature vector: basic stats + COP stats + step metrics + spectral energy"""
    base = extract_gait_features(arr)

    cop = compute_cop_per_sample(arr)
    # stats for COP arrays
    cop_feats = []
    for k in ('l_x','l_y','r_x','r_y'):
        a = cop[k]
        cop_feats.extend([
            np.nanmean(a), np.nanstd(a), np.nanmin(a), np.nanmax(a)
        ])

    # steps
    step_count, mean_interval = detect_steps_from_total_force(arr, sr=sr)
    cadence = np.nan
    duration = (arr[-1,0] - arr[0,0]) if arr.shape[0] > 1 else np.nan
    if duration and not np.isnan(duration) and duration>0:
        cadence = (step_count / duration) * 60.0

    # simple spectral energy of combined total force
    try:
        total = np.nansum(arr[:, 1:17], axis=1)
        # remove nan
        total = np.nan_to_num(total)
        f = np.fft.rfft(total - np.mean(total))
        spec_energy = np.sum(np.abs(f)**2)
    except Exception:
        spec_energy = 0.0

    extra = np.array(cop_feats + [step_count, mean_interval if not np.isnan(mean_interval) else -1.0, cadence if not np.isnan(cadence) else -1.0, spec_energy], dtype=np.float32)
    return np.concatenate([base.astype(np.float32), extra])

def walk_gait_dir(data_dir):
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.txt'):
            continue
        label = 'patient' if 'Pt' in fname or 'Pt' in fname else 'healthy'
        # more robust: filenames containing 'Pt' indicate patient, 'Co' control/healthy
        if 'Pt' in fname or fname.startswith('GaPt') or fname.startswith('JuPt') or fname.startswith('SiPt'):
            label = 'patient'
        else:
            label = 'healthy'
        yield os.path.join(data_dir, fname), label

def load_image(path, size=(128,128)):
    img = Image.open(path).convert('L')
    img = img.resize(size)
    arr = np.array(img).astype(np.float32)/255.0
    return arr

def extract_image_features(arr):
    # arr: HxW grayscale in [0,1]
    hist, _ = np.histogram(arr, bins=32, range=(0,1))
    return hist.astype(np.float32)

def walk_images_dir(data_dir):
    # expects subfolders `healthy` and `patient` (or similar)
    for label in ['healthy','patient']:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    yield os.path.join(root,f), label

def walk_motor_preprocessed(data_dir):
    # expects a preprocessed folder with observation_*.json and patient_*.json
    pre = os.path.join(data_dir, 'preprocessed')
    if not os.path.isdir(pre):
        # maybe data_dir is already preprocessed folder
        pre = data_dir
    for fname in os.listdir(pre):
        if fname.endswith('.json') and fname.startswith('observation'):
            yield os.path.join(pre, fname)

def load_motor_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def extract_motor_features(js):
    # js: a dict with timeseries fields; we'll collect summary stats for numeric arrays found
    feats = []
    def collect(obj):
        if isinstance(obj, list):
            arr = np.array([x for x in obj if isinstance(x,(int,float))])
            if arr.size>0:
                feats.extend([arr.mean(), arr.std(), arr.min(), arr.max()])
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)
    collect(js)
    if len(feats)==0:
        return np.zeros(4, dtype=np.float32)
    return np.array(feats, dtype=np.float32)
