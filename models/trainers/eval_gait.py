"""Evaluate a saved gait model across a gait dataset folder.

Saves:
- gait_classification_report.txt
- gait_per_file_predictions.csv
- gait_confusion_matrix.png

Usage:
.venv\Scripts\python.exe models/trainers/eval_gait.py --model models/trained_models/gait_signals/model_gait_enhanced.joblib --data-dir "new_data/Gait Signals/gait-in-parkinsons-disease-1.0.0" --out models/trained_models/gait_signals
"""
import argparse
import os
import re
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from models.utils.dataset import load_gait_file, extract_gait_features, extract_gait_features_enhanced


def infer_label_from_filename(fname):
    # reuse heuristics from train_gait.py
    m = re.search(r'(?i)(?:^|[^a-zA-Z])(pt|co)(?:[^a-zA-Z]|$)', fname)
    if m:
        lbl = m.group(1).lower()
        return 1 if lbl == 'pt' else 0
    low = fname.lower()
    if 'pt' in low:
        return 1
    if 'co' in low:
        return 0
    return None


def pick_extractor(clf, sample_arr):
    # try to choose enhanced vs base based on feature length or try/predict
    en_feats = None
    base_feats = None
    try:
        en_feats = extract_gait_features_enhanced(sample_arr)
    except Exception:
        en_feats = None
    try:
        base_feats = extract_gait_features(sample_arr)
    except Exception:
        base_feats = None

    if hasattr(clf, 'n_features_in_'):
        n = clf.n_features_in_
        if en_feats is not None and getattr(en_feats, 'size', None) == n:
            return extract_gait_features_enhanced
        if base_feats is not None and getattr(base_feats, 'size', None) == n:
            return extract_gait_features

    # fallback: try predict_proba
    if en_feats is not None:
        try:
            clf.predict_proba(en_feats.reshape(1, -1))
            return extract_gait_features_enhanced
        except Exception:
            pass
    if base_feats is not None:
        try:
            clf.predict_proba(base_feats.reshape(1, -1))
            return extract_gait_features
        except Exception:
            pass
    return extract_gait_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    clf = joblib.load(args.model)
    files = [f for f in sorted(os.listdir(args.data_dir)) if f.lower().endswith('.txt')]
    if not files:
        print('No gait .txt files found in', args.data_dir)
        return

    # choose extractor based on first file
    sample_arr = load_gait_file(os.path.join(args.data_dir, files[0]))
    extractor = pick_extractor(clf, sample_arr)

    rows = []
    y_true = []
    y_pred = []
    for f in files:
        path = os.path.join(args.data_dir, f)
        try:
            arr = load_gait_file(path)
            feats = extractor(arr).reshape(1, -1)
            pred = clf.predict(feats)[0]
            proba = clf.predict_proba(feats)[0].max()
            true = infer_label_from_filename(f)
            rows.append({'filename': f, 'pred': int(pred), 'proba': float(proba), 'true': (int(true) if true is not None else '')})
            if true is not None:
                y_true.append(int(true))
                y_pred.append(int(pred))
        except Exception as e:
            print('failed', f, e)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'gait_per_file_predictions.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if len(y_true) == 0:
        print('No ground-truth labels could be inferred from filenames; predictions written to', out_csv)
        return

    report = classification_report(y_true, y_pred)
    rpt_path = os.path.join(out_dir, 'gait_classification_report.txt')
    with open(rpt_path, 'w', encoding='utf8') as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Gait confusion matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.tight_layout()
    cm_path = os.path.join(out_dir, 'gait_confusion_matrix.png')
    fig.savefig(cm_path)

    print('Saved report to', rpt_path)
    print('Saved predictions CSV to', out_csv)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()
