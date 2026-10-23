"""Evaluate a trained motor RandomForest model and generate proof artifacts.

Usage example:
  .venv\Scripts\python.exe -u models.trainers.eval_motor --model models/trained_models/motor_baseline_full/motor_rf_baseline.joblib \
    --features models/trained_models/motor_baseline_full/motor_features_per_subject.csv --out models/trained_models/motor_baseline_full

This will write the classification report (text), per-subject predictions CSV, and confusion matrix PNG into the output folder.
"""

from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to joblib model produced by train_motor.py')
    p.add_argument('--features', required=True, help='Per-subject features CSV produced by train_motor.py')
    p.add_argument('--out', required=True, help='Output directory where proofs will be written')
    p.add_argument('--label-map', required=False, help='Optional path to preprocessed/file_list.csv to map subject ids to labels')
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    feat_csv = Path(args.features)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = joblib.load(model_path)
    # model artifact saved as dict {'model': clf, 'features': [...]}
    if isinstance(m, dict) and 'model' in m:
        clf = m['model']
        cols = m.get('features')
    else:
        clf = m
        cols = None

    df = pd.read_csv(feat_csv, index_col=0)
    # if labels are not present, try to load mapping from preprocessed/file_list.csv
    if 'label' not in df.columns:
        # priority: provided label_map, else try default dataset path used by trainer
        label_map_path = None
        if args.label_map:
            label_map_path = Path(args.label_map)
        else:
            default_root = Path(r"d:\Bunny\ML_mini_project\new_data\Motor Tasks\pads-parkinsons-disease-smartwatch-dataset-1.0.0")
            candidate = default_root / 'preprocessed' / 'file_list.csv'
            if candidate.exists():
                label_map_path = candidate

        if label_map_path is None or not label_map_path.exists():
            raise RuntimeError('features CSV lacks a "label" column and no label map found. Provide --label-map pointing to preprocessed/file_list.csv')

        meta = pd.read_csv(label_map_path, header=None)
        # file_list.csv in this dataset has patient id in a column; try to find id and label columns
        # attempt common formats: headerless with id in column 1 and label in last column
        if meta.shape[1] >= 2:
            # try heuristics: if headerless, assume id is column 0 or 1
            # many earlier reads used column 'id' and 'label' with headers; try to re-read with header=0
            meta2 = pd.read_csv(label_map_path)
            if 'id' in meta2.columns and 'label' in meta2.columns:
                id2label = dict(zip(meta2['id'].astype(str), meta2['label'].astype(int)))
            else:
                # fallback: assume first column contains id-like strings and final column is numeric label
                idcol = meta2.columns[0]
                labcol = meta2.columns[-1]
                id2label = dict(zip(meta2[idcol].astype(str), meta2[labcol].astype(int)))
        else:
            raise RuntimeError(f'Unexpected format for label map at {label_map_path}')

        # map df index (subject_id) to label
        df['label'] = df.index.astype(str).map(id2label)

    X = df.drop(columns=['label'])
    y = df['label']

    if cols is not None:
        # ensure columns order matches model feature order
        X = X.reindex(columns=cols)

    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, output_dict=False)
    report_dict = classification_report(y, y_pred, output_dict=True)

    # save text report
    rpt_path = out_dir / 'motor_classification_report.txt'
    with rpt_path.open('w', encoding='utf-8') as f:
        f.write(report)

    # per-subject predictions
    pred_df = pd.DataFrame({'subject_id': df.index.astype(str), 'label_true': y, 'label_pred': y_pred})
    pred_csv = out_dir / 'motor_per_subject_predictions.csv'
    pred_df.to_csv(pred_csv, index=False)

    # confusion matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('Confusion matrix (all subjects)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    fig.tight_layout()
    cm_path = out_dir / 'motor_confusion_matrix.png'
    fig.savefig(cm_path)

    # summary
    print('Saved classification report to', rpt_path)
    print('Saved per-subject predictions to', pred_csv)
    print('Saved confusion matrix to', cm_path)


if __name__ == '__main__':
    main()
