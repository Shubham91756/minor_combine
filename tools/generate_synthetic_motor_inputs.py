#!/usr/bin/env python3
"""Generate synthetic Motor inputs for the demo UI/API.

Why this exists:
- The UI's Motor CSV picker only accepts `.csv`, and the Motor timeseries picker only accepts `.txt`.
- The raw dataset also contains many `.json` files (observation_*.json) that are NOT accepted by the UI.
- This script creates small, valid example files so you can test the end-to-end pipeline.

Outputs (created under sample_data/motor_tasks/synthetic/):
- synthetic_motor_features.csv  (tabular features; one row per sample)
- synthetic_motor_features.json (single-sample feature dict)
- synthetic_healthy_timeseries.txt
- synthetic_patient_timeseries.txt

Note:
These are *synthetic* and only for demo/testing connectivity. They are not clinically meaningful.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def _load_feature_names(repo_root: Path) -> list[str]:
    model_path = repo_root / 'models' / 'trained_models' / 'motor_baseline_full' / 'motor_rf_further_retrained.joblib'
    art = joblib.load(model_path)
    if isinstance(art, dict) and 'features' in art:
        return list(art['features'])
    # fallback: infer from existing feature csv
    feat_csv = repo_root / 'models' / 'trained_models' / 'motor_baseline_full' / 'motor_features_per_subject_recomputed.csv'
    df = pd.read_csv(feat_csv, index_col=0)
    return list(df.columns)


def _fit_reference_distribution(repo_root: Path, feature_names: list[str]) -> pd.DataFrame:
    feat_csv = repo_root / 'models' / 'trained_models' / 'motor_baseline_full' / 'motor_features_per_subject_recomputed.csv'
    if feat_csv.exists():
        df = pd.read_csv(feat_csv, index_col=0)
        df = df.reindex(columns=feature_names)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return df
    # if not present, fall back to simple defaults
    return pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)


def _sample_features(ref: pd.DataFrame, n: int, kind: str, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic tabular motor features.

    `kind` is one of: healthy, patient, other.

    We sample around observed mean/std and apply a mild heuristic shift:
    - patient: slightly higher std/rms/energy features
    - other: moderate shift
    """
    mu = ref.mean(axis=0)
    sigma = ref.std(axis=0).replace(0, 1e-6)

    # clip bounds based on robust quantiles
    lo = ref.quantile(0.01)
    hi = ref.quantile(0.99)

    Z = rng.standard_normal((n, ref.shape[1]))
    X = (Z * sigma.values) + mu.values

    # Heuristic shifts
    mult = 1.0
    if kind == 'patient':
        mult = 1.12
    elif kind == 'other':
        mult = 1.06

    cols = list(ref.columns)
    for j, c in enumerate(cols):
        cl = c.lower()
        if any(k in cl for k in ('_std', '_rms', '_energy', 'acc_mag_std')):
            X[:, j] *= mult

    # final clip
    X = np.minimum(np.maximum(X, lo.values), hi.values)
    out = pd.DataFrame(X, columns=cols)

    # ensure non-negative for known non-negative features
    for c in cols:
        if any(k in c.lower() for k in ('energy', 'rms', 'ptp', 'nrows', 'duration', 'median_dt')):
            out[c] = out[c].clip(lower=0.0)

    return out


def _write_timeseries(path: Path, rng: np.random.Generator, patient_like: bool) -> None:
    """Create a simple 7-col timeseries: time + 6 channels."""
    n = 1200
    dt = 0.01
    t = np.arange(n, dtype=np.float32) * dt

    # healthy: smoother; patient_like: more tremor noise
    base_freq = 0.8
    tremor_freq = 6.0 if patient_like else 2.0
    tremor_amp = 0.9 if patient_like else 0.25

    channels = []
    for i in range(6):
        phase = rng.uniform(0, 2 * np.pi)
        base = 0.4 * np.sin(2 * np.pi * base_freq * t + phase)
        tremor = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + phase)
        noise = rng.normal(0.0, 0.08 if patient_like else 0.03, size=n)
        ch = base + tremor + noise
        channels.append(ch)

    arr = np.column_stack([t] + channels)

    # Write comma-separated, no header (matches trainer's robust loader)
    with path.open('w', encoding='utf-8') as f:
        for row in arr:
            f.write(','.join(f'{float(x):.6f}' for x in row) + '\n')


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / 'sample_data' / 'motor_tasks' / 'synthetic'
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    feature_names = _load_feature_names(repo_root)
    ref = _fit_reference_distribution(repo_root, feature_names)

    # build small set
    df_healthy = _sample_features(ref, n=2, kind='healthy', rng=rng)
    df_patient = _sample_features(ref, n=2, kind='patient', rng=rng)
    df_other = _sample_features(ref, n=1, kind='other', rng=rng)

    df = pd.concat([df_healthy, df_patient, df_other], ignore_index=True)
    df.insert(0, 'sample_id', [f'synth_{i:02d}' for i in range(len(df))])

    csv_path = out_dir / 'synthetic_motor_features.csv'
    df.to_csv(csv_path, index=False)

    # also write a single JSON feature dict using first row (excluding sample_id)
    json_path = out_dir / 'synthetic_motor_features.json'
    payload = {k: float(v) for k, v in df.drop(columns=['sample_id']).iloc[0].to_dict().items()}
    json_path.write_text(json.dumps({'features': payload}, indent=2), encoding='utf-8')

    # timeseries
    _write_timeseries(out_dir / 'synthetic_healthy_timeseries.txt', rng=rng, patient_like=False)
    _write_timeseries(out_dir / 'synthetic_patient_timeseries.txt', rng=rng, patient_like=True)

    print('Wrote:')
    for p in [csv_path, json_path, out_dir / 'synthetic_healthy_timeseries.txt', out_dir / 'synthetic_patient_timeseries.txt']:
        print(' -', p)


if __name__ == '__main__':
    main()
