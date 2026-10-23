"""Train a simple classifier on the gait text files.

Usage:
  python -m models.trainers.train_gait --data-dir "..." --out model_gait.joblib
"""
import argparse
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from models.utils.dataset import load_gait_file, extract_gait_features, extract_gait_features_enhanced
import re

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--enhanced', action='store_true', help='Use enhanced gait features (COP, step metrics, spectral energy)')
    p.add_argument('--out', default='model_gait.joblib')
    args = p.parse_args()

    X = []
    y = []
    skipped = 0
    # iterate files in the folder and infer label from filename
    for fname in sorted(os.listdir(args.data_dir)):
        if not fname.lower().endswith('.txt'):
            continue
        # skip obvious metadata files
        if any(fname.lower().startswith(prefix) for prefix in ('demographics', 'format', 'sha256', 'gaitpd')):
            continue
        path = os.path.join(args.data_dir, fname)
        # infer label: look for 'Pt' (patient) or 'Co' (control) in the filename
        m = re.search(r'(?i)(?:^|[^a-zA-Z])(pt|co)(?:[^a-zA-Z]|$)', fname)
        if m:
            lbl = m.group(1).lower()
            label = 'patient' if lbl == 'pt' else 'control'
        else:
            # fallback heuristics
            if 'pt' in fname.lower():
                label = 'patient'
            elif 'co' in fname.lower():
                label = 'control'
            else:
                print('skip (unknown label)', fname)
                skipped += 1
                continue

        try:
            arr = load_gait_file(path)
            if args.enhanced:
                feats = extract_gait_features_enhanced(arr)
            else:
                feats = extract_gait_features(arr)
            X.append(feats)
            y.append(1 if label == 'patient' else 0)
        except Exception as e:
            print('skip', path, e)
            skipped += 1

    if len(X) == 0:
        print('No valid gait files found in', args.data_dir)
        return

    print('loaded samples:', len(X), 'skipped:', skipped)

    X = np.stack(X)
    y = np.array(y)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test,y_pred))

    joblib.dump(clf, args.out)
    print('saved', args.out)

if __name__=='__main__':
    main()
