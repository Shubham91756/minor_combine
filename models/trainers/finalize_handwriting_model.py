"""Finalize handwriting model by training on the full dataset and saving scaler + classifier.

This script will:
- extract embeddings for every image listed in the provided data-list
- fit a StandardScaler on those embeddings
- train a final soft-voting ensemble (LogisticRegression + RandomForest with tuned params)
- save a joblib containing {'model': ensemble, 'scaler': scaler, 'class_names': class_names}

Usage:
.venv\Scripts\python.exe models/trainers/finalize_handwriting_model.py --model models/trained_models/handwriting_efficient_cpu_improved/handwriting_embed_model.joblib --data-list D:\...\handwriting_all_valid_images.txt --out models/trained_models/handwriting_efficient_cpu_improved
"""
from pathlib import Path
import argparse
import joblib
import numpy as np


def build_embed_model(img_size=224):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    base = tf.keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
    model = keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.BatchNormalization()])
    return model


def preprocess_image_tensor(path, img_size):
    import tensorflow as tf
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def extract_embeddings(paths, embed_model, img_size, batch_size=32):
    import tensorflow as tf
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: preprocess_image_tensor(p, img_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    embs = []
    for batch in ds:
        e = embed_model(batch, training=False).numpy()
        embs.append(e)
    if embs:
        return np.vstack(embs)
    return np.zeros((0, embed_model.output_shape[-1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-list', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(str(model_path))
    class_names = artifact.get('class_names', None) if isinstance(artifact, dict) else None

    data_list = Path(args.data_list)
    paths = [line.strip() for line in data_list.read_text(encoding='utf8').splitlines() if line.strip()]
    print('Extracting embeddings for', len(paths), 'images...')
    embed_model = build_embed_model(img_size=args.img_size)
    X = extract_embeddings(paths, embed_model, args.img_size, batch_size=args.batch_size)

    # fit scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # train final ensemble (LogReg + RF)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    # simple RF params based on tuning earlier
    rf = RandomForestClassifier(n_estimators=800, min_samples_split=5, max_depth=None, class_weight='balanced', n_jobs=-1, random_state=1337)
    lr = LogisticRegression(max_iter=1000)
    ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')

    # infer labels from file paths same as train script
    def get_class_from_path(p):
        parts = Path(p).parts
        for i, part in enumerate(parts):
            if part.lower() == 'handwriting' and i + 1 < len(parts):
                return parts[i + 1]
        return Path(p).parents[1].name

    y = np.array([0 if get_class_from_path(p) == class_names[0] else 1 for p in paths]) if class_names is not None else None
    if y is None:
        raise SystemExit('class_names not found in model artifact; cannot infer labels to retrain on full set')

    print('Training final ensemble on full dataset...')
    ensemble.fit(Xs, y)

    final_art = {'model': ensemble, 'scaler': scaler, 'class_names': class_names}
    joblib.dump(final_art, str(out_dir / 'handwriting_embed_model_final.joblib'))
    print('Saved final model to', out_dir / 'handwriting_embed_model_final.joblib')


if __name__ == '__main__':
    main()
