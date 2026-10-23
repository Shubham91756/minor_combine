"""Re-extract features from motor timeseries, train RandomForest with upsampling, and save artifacts.

Writes to the specified out directory:
- motor_features_per_subject_recomputed.csv
- motor_rf_improved.joblib
- predictions_improved.csv

Usage (from repo root):
.venv\Scripts\python.exe models/trainers/train_motor_reextract.py --timeseries-dir "new_data/.../movement/timeseries" --label-map "new_data/.../preprocessed/file_list.csv" --out "models/trained_models/motor_baseline_full"
"""
import argparse
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib


def file_features(path):
    # read with comma sep; fallback to whitespace
    try:
        df = pd.read_csv(path, sep=',', header=None)
    except Exception:
        df = pd.read_csv(path, sep='\s+', header=None)
    # expect first column time, next 6 columns are signals
    # if only one column read, try to split by comma manually
    if df.shape[1] == 1:
        df = df[0].str.split(',', expand=True).astype(float)
    # convert to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    # compute features per channel (cols 1..6)
    out = {}
    ncols = df.shape[1]
    sig_cols = list(range(1, min(7, ncols)))  # 1..6 if present
    # if only time + 6 signals expected, ensure we have them
    for i in sig_cols:
        col = df.iloc[:, i].dropna()
        prefix = f'c{i}'
        out[f'{prefix}_mean'] = col.mean()
        out[f'{prefix}_std'] = col.std()
        out[f'{prefix}_median'] = col.median()
        out[f'{prefix}_min'] = col.min()
        out[f'{prefix}_max'] = col.max()
        out[f'{prefix}_ptp'] = col.max() - col.min() if not col.empty else np.nan
        out[f'{prefix}_rms'] = np.sqrt((col ** 2).mean()) if not col.empty else np.nan
        out[f'{prefix}_energy'] = (col ** 2).sum()
    # magnitude features if at least 4..6 channels (acc 1-3)
    if len(sig_cols) >= 3:
        acc = df.iloc[:, 1:4].astype(float).fillna(0).values
        mag = np.linalg.norm(acc, axis=1)
        out['acc_mag_mean'] = mag.mean()
        out['acc_mag_std'] = mag.std()
    else:
        out['acc_mag_mean'] = np.nan
        out['acc_mag_std'] = np.nan

    # time column duration and median dt
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    out['nrows'] = int(df.shape[0])
    out['duration'] = float(t.max() - t.min()) if t.notnull().any() else np.nan
    dts = t.diff().dropna()
    out['median_dt'] = float(dts.median()) if not dts.empty else np.nan
    return out


def aggregate_per_subject(timeseries_dir):
    files = sorted(glob.glob(os.path.join(timeseries_dir, '*.txt')))
    per_file = []
    for f in files:
        fn = os.path.basename(f)
        subject = fn.split('_', 1)[0].lstrip('0')
        feats = file_features(f)
        feats['file'] = fn
        feats['subject'] = subject
        per_file.append(feats)
    pf = pd.DataFrame(per_file)
    # aggregate per-subject by mean for numeric columns
    numeric_cols = pf.select_dtypes(include=[np.number]).columns.tolist()
    agg = pf.groupby('subject')[numeric_cols].mean().reset_index()
    # ensure subject as index (string of integer without padding)
    agg['subject'] = agg['subject'].astype(str).str.lstrip('0')
    agg = agg.set_index('subject')
    return agg, pf


def load_label_map(label_map_path):
    # robustly parse label map
    p = Path(label_map_path)
    if not p.exists():
        raise FileNotFoundError(label_map_path)
    try:
        df = pd.read_csv(p)
        if 'id' in df.columns and 'label' in df.columns:
            df['id_norm'] = df['id'].astype(str).str.lstrip('0')
            df['id_norm'] = pd.to_numeric(df['id_norm'], errors='coerce').astype('Int64')
            df = df[['id_norm', 'label']].dropna()
            df['id_norm'] = df['id_norm'].astype(int)
            id2label = dict(zip(df['id_norm'].astype(str), df['label'].astype(int)))
            return id2label
    except Exception:
        pass
    # fallback headerless
    meta = pd.read_csv(p, header=None)
    if meta.shape[1] >= 2:
        ids = meta.iloc[:, 0].astype(str).str.lstrip('0')
        labels = meta.iloc[:, -1]
        id2label = dict(zip(ids, labels.astype(int)))
        return id2label
    raise RuntimeError('Could not parse label map')


def upsample_training(X_train, y_train, random_state=42):
    # simple upsampling of minority classes to match majority class size
    df = X_train.copy()
    df['label'] = y_train.values
    counts = df['label'].value_counts()
    max_n = counts.max()
    resampled = []
    for lab, grp in df.groupby('label'):
        if len(grp) < max_n:
            up = resample(grp, replace=True, n_samples=max_n, random_state=random_state)
            resampled.append(up)
        else:
            resampled.append(grp)
    big = pd.concat(resampled).sample(frac=1, random_state=random_state)
    y = big['label']
    X = big.drop(columns=['label'])
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--timeseries-dir', required=True)
    p.add_argument('--label-map', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--features-csv', required=False, help='If provided, skip extraction and load precomputed features CSV')
    p.add_argument('--augment', action='store_true', help='Generate synthetic healthy samples and append to training data')
    p.add_argument('--synth-count', type=int, default=2000, help='Number of synthetic healthy samples to generate')
    p.add_argument('--synth-method', choices=['independent','mv'], default='independent', help='Synthesis method: independent per-feature or multivariate')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load or extract features
    if args.features_csv:
        features_csv = Path(args.features_csv)
        print('Loading precomputed features from', features_csv)
        agg = pd.read_csv(features_csv, index_col=0)
        # ensure index is subject
        agg.index = agg.index.astype(str).str.lstrip('0')
        # per_file unknown in this mode
        per_file = pd.DataFrame()
    else:
        print('Scanning and extracting features...')
        agg, per_file = aggregate_per_subject(args.timeseries_dir)
        features_csv = out_dir / 'motor_features_per_subject_recomputed.csv'
        agg.to_csv(features_csv, index=True)
        print('Wrote features to', features_csv)

    print('Loading labels...')
    # load_label_map can accept different CSV shapes; if it's the predictions file, handle specially
    label_map_path = Path(args.label_map)
    if label_map_path.exists():
        try:
            tmp = pd.read_csv(label_map_path)
            if 'subject_id' in tmp.columns and 'label_true' in tmp.columns:
                id2label = dict(zip(tmp['subject_id'].astype(str).str.lstrip('0'), tmp['label_true'].astype(int)))
            else:
                id2label = load_label_map(label_map_path)
        except Exception:
            id2label = load_label_map(label_map_path)
    else:
        raise FileNotFoundError(args.label_map)
    # map labels into features
    agg2 = agg.copy()
    agg2['subject_id'] = agg2.index.astype(str).str.lstrip('0')
    agg2['label'] = agg2['subject_id'].map(id2label)
    missing = agg2['label'].isnull().sum()
    if missing > 0:
        print(f'Warning: {missing} subjects have no label mapping')
    agg2 = agg2.drop(columns=['subject_id'])

    # prepare X,y
    X = agg2.drop(columns=['label']).fillna(0)
    y = agg2['label'].astype(int)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # optionally generate synthetic healthy samples and append to training set
    if args.augment:
        # create class-0 frame
        class0_mask = (y_train == 0)
        if class0_mask.sum() == 0:
            print('No class-0 rows in training set to base synthesis on; skipping augmentation')
        else:
            X0 = X_train[class0_mask]
            n_synth = args.synth_count
            print(f'Generating {n_synth} synthetic healthy samples using method={args.synth_method}...')
            if args.synth_method == 'independent':
                means = X0.mean()
                stds = X0.std().replace(0, 1e-6)
                lo = X0.quantile(0.01)
                hi = X0.quantile(0.99)
                synth = np.random.randn(n_synth, X0.shape[1]) * stds.values + means.values
                synth = np.clip(synth, lo.values, hi.values)
                synth = pd.DataFrame(synth, columns=X0.columns)
            else:
                # multivariate normal with regularized covariance
                mu = X0.mean().values
                cov = np.cov(X0.values, rowvar=False)
                # regularize
                eps = 1e-6 * np.eye(cov.shape[0])
                cov += eps
                try:
                    synth_vals = np.random.multivariate_normal(mean=mu, cov=cov, size=n_synth)
                except Exception:
                    # fallback to independent
                    means = X0.mean()
                    stds = X0.std().replace(0, 1e-6)
                    synth_vals = np.random.randn(n_synth, X0.shape[1]) * stds.values + means.values
                lo = X0.quantile(0.01).values
                hi = X0.quantile(0.99).values
                synth_vals = np.minimum(np.maximum(synth_vals, lo), hi)
                synth = pd.DataFrame(synth_vals, columns=X0.columns)
            # append to training set
            X_train = pd.concat([X_train, synth], ignore_index=True)
            y_train = pd.concat([y_train, pd.Series([0]*len(synth))], ignore_index=True)

    # upsample training set
    X_train_up, y_train_up = upsample_training(X_train, y_train)

    print('Training RandomForest...')
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train_up, y_train_up)

    # save model and feature order
    model_out = out_dir / 'motor_rf_retrained.joblib'
    joblib.dump({'model': clf, 'features': list(X.columns)}, model_out)
    print('Saved model to', model_out)

    # predictions on all subjects
    y_pred = clf.predict(X)
    pred_df = pd.DataFrame({'subject_id': X.index.astype(str), 'pred': y_pred, 'true': y.values})
    pred_csv = out_dir / 'predictions_improved.csv'
    pred_df.to_csv(pred_csv, index=False)
    print('Wrote predictions to', pred_csv)

    # also write per-file features for debugging
    per_file_csv = out_dir / 'motor_per_file_features.csv'
    per_file.to_csv(per_file_csv, index=False)
    print('Wrote per-file features to', per_file_csv)


if __name__ == '__main__':
    main()
