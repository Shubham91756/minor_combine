"""EfficientNet trainer for handwriting classification.

Features:
- Uses EfficientNetB0 as backbone (imagenet weights).
- Pure tf.data pipeline using validated image list.
- Augmentation via Keras preprocessing layers.
- Options: oversample, class_weight, focal loss, cosine decay lr schedule.
- Two-stage training: head then optional fine-tune (unfreeze last N layers).
- Saves checkpoints, SavedModel, metadata and confusion matrix.

Run smoke test:
  .venv\Scripts\python.exe -m models.trainers.train_handwriting_efficient --data-list models/trained_models/handwriting_all_valid_images.txt --out models/trained_models/handwriting_efficient_run --epochs-head 1 --epochs-ft 0
"""
from pathlib import Path
import argparse
import json
import random


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data-list', required=True)
    p.add_argument('--out', default='models/trained_models/handwriting_efficient')
    p.add_argument('--mode', choices=['end2end', 'embed'], default='embed',
                   help='Training mode: end2end trains the full Keras model; embed extracts embeddings and trains a tabular classifier (faster/better for small datasets)')
    p.add_argument('--mixed-precision', action='store_true', help='Enable TF mixed precision (mixed_float16) for faster GPU training')
    p.add_argument('--use-distribute', action='store_true', help='Use tf.distribute.MirroredStrategy when multiple GPUs are available')
    p.add_argument('--use-gpu', action='store_true', help='Allow use of GPU; by default the script will run on CPU')
    p.add_argument('--img-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs-head', type=int, default=4)
    p.add_argument('--epochs-ft', type=int, default=8)
    p.add_argument('--unfreeze-n', type=int, default=30)
    p.add_argument('--lr-head', type=float, default=1e-4)
    p.add_argument('--lr-ft', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--use-class-weight', action='store_true')
    p.add_argument('--oversample', action='store_true')
    p.add_argument('--use-lgb', action='store_true', help='Use LightGBM for classifier when in embed mode (recommended)')
    p.add_argument('--use-focal', action='store_true')
    p.add_argument('--focal-gamma', type=float, default=2.0)
    p.add_argument('--use-cosine', action='store_true')
    p.add_argument('--mixup', action='store_true', help='Apply mixup augmentation to training batches (embed mode will still extract from augmented images)')
    p.add_argument('--embed-cv', action='store_true', help='Run k-fold CV on embeddings with LightGBM (fast, high-accuracy)')
    p.add_argument('--cv-folds', type=int, default=5, help='Number of folds for embed CV')
    p.add_argument('--use-smote', action='store_true', help='Apply SMOTE to embeddings before training (requires imbalanced-learn)')
    p.add_argument('--rf-tune', action='store_true', help='Run a small randomized search for RandomForest hyperparameters when LightGBM is not available')
    p.add_argument('--try-ensemble', action='store_true', help='Build a soft-voting ensemble (LogReg + RF) as a fallback')
    return p


def main(args):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.model_selection import train_test_split
    from sklearn.utils import class_weight
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # GPU / mixed precision setup
    # Default: force CPU unless --use-gpu is provided. This makes behavior deterministic for CPU-only training.
    try:
        gpus = tf.config.list_physical_devices('GPU')
    except Exception:
        gpus = []
    if not args.use_gpu:
        # hide GPUs from TF to force CPU execution
        try:
            if gpus:
                tf.config.set_visible_devices([], 'GPU')
                print('Forcing CPU: GPUs hidden from TensorFlow')
        except Exception:
            pass
        print('Running on CPU')
    else:
        if gpus:
            print(f'GPUs visible: {[g.name for g in gpus]}')
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        else:
            print('No GPUs detected; running on CPU')

    if args.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print('Enabled mixed precision (mixed_float16)')
        except Exception as e:
            print('Mixed precision could not be enabled:', e)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_list = Path(args.data_list)
    if not data_list.exists():
        raise SystemExit('data list not found: ' + str(data_list))
    paths = [line.strip() for line in data_list.read_text(encoding='utf8').splitlines() if line.strip()]
    if not paths:
        raise SystemExit('no image paths found in data list')

    def get_class_from_path(p):
        parts = Path(p).parts
        for i, part in enumerate(parts):
            if part.lower() == 'handwriting' and i + 1 < len(parts):
                return parts[i + 1]
        return Path(p).parents[1].name

    labels = [get_class_from_path(p) for p in paths]
    class_names = sorted(list(set(labels)))
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    y = [name_to_idx[n] for n in labels]

    print('Found', len(paths), 'paths, classes=', class_names)

    # oversample optional
    if args.oversample:
        from collections import Counter
        cnt = Counter(y)
        maxc = max(cnt.values())
        new_paths = []
        new_y = []
        for label in cnt:
            group = [p for p, yy in zip(paths, y) if yy == label]
            if not group:
                continue
            repeats = maxc // len(group)
            rem = maxc % len(group)
            new_paths.extend(group * repeats)
            new_paths.extend(random.sample(group, rem))
            new_y.extend([label] * (len(group) * repeats + rem))
        paths = new_paths
        y = new_y
        print('After oversample total paths:', len(paths))

    train_files, val_files, train_labels, val_labels = train_test_split(paths, y, test_size=0.2, stratify=y, random_state=args.seed)

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (args.img_size, args.img_size))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(buffer_size=len(train_files), seed=args.seed)
    train_ds = train_ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=AUTOTUNE)
    # augmentation
    augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.15),
        layers.RandomTranslation(0.08, 0.08),
    ], name='augmentation')
    def apply_aug(img, lab):
        # Some augmentation ops (e.g. RandomContrast) only have float32 CPU kernels.
        # Ensure we run augmentation in float32 to avoid "kernel not found for DT_HALF" errors
        # when mixed precision (mixed_float16) is enabled on a CPU-only machine.
        if img.dtype != tf.float32:
            img32 = tf.cast(img, tf.float32)
        else:
            img32 = img
        # Temporarily force float32 policy for augmentation to avoid float16-only op issues on CPU
        try:
            from tensorflow.keras import mixed_precision
            old_policy = mixed_precision.global_policy().name
            mixed_precision.set_global_policy('float32')
            aug = augmentation(img32, training=True)
        finally:
            try:
                mixed_precision.set_global_policy(old_policy)
            except Exception:
                # If mixed_precision isn't available or resetting fails, proceed without raising
                pass
        # keep output as float32 (the model will handle casting if needed)
        return aug, lab

    train_ds = train_ds.map(apply_aug, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(args.batch_size).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(lambda p, l: preprocess_image(p, l), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(AUTOTUNE)

    # model - create inside strategy.scope() if using distribution
    strategy = None
    if args.use_distribute and len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print('Using MirroredStrategy with', strategy.num_replicas_in_sync, 'replicas')

    def build_models():
        # When mixed precision is enabled the global policy may be 'mixed_float16'
        # which can cause some keras application ops to receive float16 tensors while
        # constants (e.g. IMAGENET_STDDEV_RGB) remain float32, leading to dtype
        # mismatch errors. Build the backbone under float32 and then restore the
        # mixed precision policy so the rest of the model can benefit from it.
        try:
            from tensorflow.keras import mixed_precision
        except Exception:
            mixed_precision = None

        old_policy = None
        if mixed_precision is not None:
            try:
                old_policy = mixed_precision.global_policy().name
                mixed_precision.set_global_policy('float32')
            except Exception:
                old_policy = None

        base = tf.keras.applications.EfficientNetB0(input_shape=(args.img_size, args.img_size, 3), include_top=False, weights='imagenet')
        # restore mixed precision policy if it was previously set
        if mixed_precision is not None and old_policy is not None:
            try:
                mixed_precision.set_global_policy(old_policy)
            except Exception:
                pass
        base.trainable = False
        # If embed mode: we will use base model as embedding extractor
        if args.mode == 'embed':
            embed_model = keras.Sequential([base, layers.GlobalAveragePooling2D(), layers.BatchNormalization()], name='embed_model')
            print('Embedding model built; ready to extract embeddings for tabular classifier')
            return {'embed_model': embed_model, 'base': base}
        else:
            inputs = keras.Input(shape=(args.img_size, args.img_size, 3))
            x = base(inputs, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            # ensure final outputs are float32 when using mixed precision
            outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
            model = keras.Model(inputs, outputs)
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr_head), loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()
            return {'model': model, 'base': base}

    if strategy is not None:
        with strategy.scope():
            built = build_models()
    else:
        built = build_models()

    # expose built models
    embed_model = built.get('embed_model', None)
    model = built.get('model', None)
    base = built.get('base', None)

    cw = None
    if args.use_class_weight:
        unique = np.unique(train_labels)
        weights = class_weight.compute_class_weight('balanced', classes=unique, y=np.array(train_labels))
        cw = {int(c): float(w) for c, w in zip(unique, weights)}
        print('Using class weights:', cw)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(out_dir / 'best_model_head.h5'), save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    ]

    # If running in embed mode, `model` will be None (we only built an embedding extractor).
    # Skip the Keras head training when model is not present and continue to embedding extraction.
    history = None
    if model is not None:
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=callbacks, class_weight=cw)
    else:
        print('No Keras model to fit (embed mode); skipping head training and proceeding to embedding extraction')

    # fine-tune (only when we built the full Keras model)
    if base is not None and model is not None and args.unfreeze_n > 0 and args.epochs_ft > 0:
        total = len(base.layers)
        n = min(args.unfreeze_n, total)
        for layer in base.layers[: total - n]:
            layer.trainable = False
        for layer in base.layers[total - n:]:
            layer.trainable = True
        print(f'Fine-tuning: base layers total={total}, unfrozen={n}')

        # loss selection
        if args.use_focal:
            def focal_loss_fn(gamma=args.focal_gamma, alpha=0.25):
                def loss(y_true, y_pred):
                    y_true_f = tf.cast(y_true, tf.float32)
                    eps = tf.keras.backend.epsilon()
                    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
                    bce = - (y_true_f * tf.math.log(y_pred) + (1 - y_true_f) * tf.math.log(1 - y_pred))
                    p_t = y_true_f * y_pred + (1 - y_true_f) * (1 - y_pred)
                    alpha_factor = y_true_f * alpha + (1 - y_true_f) * (1 - alpha)
                    modulating_factor = tf.pow(1.0 - p_t, gamma)
                    return tf.reduce_mean(alpha_factor * modulating_factor * bce)
                return loss
            loss_fn = focal_loss_fn()
        else:
            loss_fn = 'binary_crossentropy'

        # optimizer schedule
        if args.use_cosine and args.epochs_ft > 0:
            steps_per_epoch = int(np.ceil(len(train_files) / args.batch_size))
            decay_steps = int(args.epochs_ft * steps_per_epoch)
            schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.lr_ft, decay_steps=decay_steps)
            opt = keras.optimizers.Adam(learning_rate=schedule)
            print(f'Using cosine decay lr schedule with decay_steps={decay_steps}')
        else:
            opt = keras.optimizers.Adam(learning_rate=args.lr_ft)

        model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
        # If we're using a learning rate schedule object (e.g. CosineDecay), avoid
        # adding ReduceLROnPlateau which expects numeric learning rate values and
        # may attempt arithmetic with the schedule object. Use EarlyStopping and
        # ModelCheckpoint always; add ReduceLROnPlateau only when not using a
        # schedule.
        ft_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(out_dir / 'best_model_finetune.h5'), save_best_only=True),
        ]
        if not isinstance(opt.learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            ft_callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3))
        history_ft = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft, callbacks=ft_callbacks, class_weight=cw)
        # merge histories for plotting
        if history is not None:
            for k, v in history_ft.history.items():
                history.history.setdefault(k, []).extend(v)
        else:
            history = history_ft

    # evaluate
    if args.mode == 'end2end':
        y_true = []
        y_pred = []
        for imgs, labs in val_ds:
            probs = model.predict(imgs, verbose=0)
            preds = (probs.ravel() >= 0.5).astype(int)
            y_true.extend(labs.numpy().tolist())
            y_pred.extend(preds.tolist())

        print('\nFinal classification report:')
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        cm_path = out_dir / 'confusion_efficient.png'
        fig.savefig(str(cm_path))
        print('Saved confusion matrix to', cm_path)

        model.save(str(out_dir / 'efficientnet'))
        meta = {'class_names': class_names, 'img_size': args.img_size, 'mode': args.mode}
        with open(out_dir / 'metadata.json', 'w', encoding='utf8') as f:
            json.dump(meta, f, indent=2)
        print('Saved model to', out_dir)
        # save classification report
        try:
            with open(out_dir / 'classification_report.txt', 'w', encoding='utf8') as f:
                f.write(classification_report(y_true, y_pred, target_names=class_names))
        except Exception:
            pass

        # plot training curves if available
        try:
            if history is not None:
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                # loss
                ax[0].plot(history.history.get('loss', []), label='train_loss')
                ax[0].plot(history.history.get('val_loss', []), label='val_loss')
                ax[0].set_title('Loss')
                ax[0].legend()
                # accuracy
                acc = history.history.get('accuracy', history.history.get('acc', []))
                val_acc = history.history.get('val_accuracy', history.history.get('val_acc', []))
                ax[1].plot(acc, label='train_acc')
                ax[1].plot(val_acc, label='val_acc')
                ax[1].set_title('Accuracy')
                ax[1].legend()
                plt.tight_layout()
                fig.savefig(str(out_dir / 'training_curves.png'))
                plt.close(fig)
                print('Saved training curves to', out_dir / 'training_curves.png')
        except Exception as e:
            print('Could not save training curves:', e)
        return

    # EMBED mode: extract embeddings for train & val, then train a tabular classifier
    # build embedding datasets
    def gen_ds(files_list):
        ds = tf.data.Dataset.from_tensor_slices(files_list)
        ds = ds.map(lambda p: preprocess_image(p, 0)[0], num_parallel_calls=AUTOTUNE)
        ds = ds.batch(args.batch_size).prefetch(AUTOTUNE)
        return ds

    train_embed_ds = gen_ds(train_files)
    val_embed_ds = gen_ds(val_files)

    # optionally apply mixup by generating augmented images for additional embeddings
    def extract_embeddings(ds):
        embs = []
        # iterate with tqdm for progress feedback
        for batch in tqdm(ds, desc='Extracting embeddings', unit='batch'):
            if args.mixup:
                a = augmentation(batch, training=True)
            else:
                a = batch
            e = embed_model(a, training=False).numpy()
            embs.append(e)
        if embs:
            return np.vstack(embs)
        return np.zeros((0, embed_model.output_shape[-1]))

    print('Extracting train embeddings...')
    X_train = extract_embeddings(train_embed_ds)
    print('Extracting val embeddings...')
    X_val = extract_embeddings(val_embed_ds)

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)

    print('Embeddings shapes:', X_train.shape, X_val.shape)

    # If requested, run k-fold CV on embeddings (train+val combined) and save per-fold boosters
    if args.embed_cv and args.use_lgb:
        from sklearn.model_selection import StratifiedKFold
        import lightgbm as lgb
        import joblib as _joblib

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_results = []
        boosters = []
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all)):
            print('CV fold', fold_idx + 1)
            X_tr, X_va = X_all[tr_idx], X_all[va_idx]
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05}
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)
            callbacks = [lgb.callback.early_stopping(50), lgb.callback.log_evaluation(period=0)]
            booster = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_val], callbacks=callbacks)
            boosters.append(booster)
            # evaluate
            preds = (booster.predict(X_va) >= 0.5).astype(int)
            acc = float((preds == y_va).mean())
            fold_results.append({'fold': fold_idx + 1, 'acc': acc})
            booster.save_model(str(out_dir / f'lgb_cv_fold_{fold_idx + 1}.txt'))

        # ensemble majority vote on full set
        all_preds = np.vstack([(b.predict(X_all) >= 0.5).astype(int) for b in boosters])
        maj = (all_preds.sum(axis=0) >= (len(boosters) / 2)).astype(int)
        ensemble_acc = float((maj == y_all).mean())
        print('CV fold results:', fold_results)
        print('Ensemble accuracy (on combined set):', ensemble_acc)
        _joblib.dump({'boosters': [str(out_dir / f'lgb_cv_fold_{i+1}.txt') for i in range(len(boosters))], 'fold_results': fold_results, 'ensemble_acc': ensemble_acc, 'class_names': class_names}, str(out_dir / 'embed_cv_ensemble.joblib'))
        with open(out_dir / 'embed_cv_ensemble.json', 'w', encoding='utf8') as f:
            json.dump({'fold_results': fold_results, 'ensemble_acc': ensemble_acc, 'class_names': class_names}, f, indent=2)
        print('Saved embed CV ensemble to', out_dir)
        return

    # class handling: try class_weight for lgb/rf or simple upsampling
    if args.oversample:
        # simple upsample minority
        from sklearn.utils import resample
        df = None
        try:
            import pandas as _pd
            df = _pd.DataFrame(X_train)
            df['y'] = y_train
            maxc = df['y'].value_counts().max()
            parts = [df[df['y'] == c] for c in df['y'].unique()]
            ups = [resample(p, replace=True, n_samples=maxc, random_state=args.seed) for p in parts]
            df2 = _pd.concat(ups)
            y_train = df2['y'].values
            X_train = df2.drop(columns=['y']).values
        except Exception:
            pass

    # scale embeddings
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if X_train.shape[0] > 0:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    # train classifier
    clf = None
    if args.use_lgb:
        try:
            import lightgbm as lgb
            # Use lightgbm Dataset API for stable early stopping across builds
            lgb_train = lgb.Dataset(X_train, label=y_train)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05}
            booster = lgb.train(params, lgb_train, num_boost_round=500, valid_sets=[lgb_val], early_stopping_rounds=30, verbose_eval=50)
            # wrap booster into a sklearn-like predict API
            class LGBWrapper:
                def __init__(self, booster):
                    self.booster = booster
                def predict_proba(self, X):
                    # return probability for positive class
                    probs = self.booster.predict(X)
                    # shape (n,) -> convert to (n,2) like sklearn
                    return np.vstack([1 - probs, probs]).T
                def predict(self, X):
                    probs = self.booster.predict(X)
                    return (probs >= 0.5).astype(int)
            clf = LGBWrapper(booster)
        except Exception as e:
            print('LightGBM train failed, falling back to RandomForest:', e)
            clf = None

    if clf is None:
        from sklearn.ensemble import RandomForestClassifier
        if args.rf_tune:
            try:
                from sklearn.model_selection import RandomizedSearchCV
                param_dist = {
                    'n_estimators': [200, 500, 800],
                    'max_depth': [None, 10, 20, 40],
                    'min_samples_split': [2, 5, 10]
                }
                rf = RandomForestClassifier(class_weight='balanced' if args.use_class_weight else None, n_jobs=-1, random_state=args.seed)
                rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=6, cv=3, scoring='f1', random_state=args.seed, n_jobs=-1)
                rs.fit(X_train, y_train)
                clf = rs.best_estimator_
                print('RandomForest tuned params:', rs.best_params_)
            except Exception as e:
                print('RF tuning failed, falling back to default RF:', e)
                clf = None
        if clf is None:
            clf = RandomForestClassifier(n_estimators=500, class_weight='balanced' if args.use_class_weight else None, n_jobs=-1, random_state=args.seed)
            clf.fit(X_train, y_train)

    # optional ensemble fallback
    if args.try_ensemble:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import VotingClassifier
            lr = LogisticRegression(max_iter=1000)
            # ensure rf is fit
            if not hasattr(clf, 'predict'):
                clf.fit(X_train, y_train)
            ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', clf)], voting='soft')
            ensemble.fit(X_train, y_train)
            clf = ensemble
            print('Built soft-voting ensemble (LogReg + RF)')
        except Exception as e:
            print('Could not build ensemble:', e)

    # evaluate
    # get predicted labels and probabilities
    try:
        y_proba = clf.predict_proba(X_val)
        # positive class prob
        if y_proba.ndim == 2:
            pos_proba = y_proba[:, 1]
        else:
            pos_proba = y_proba
    except Exception:
        # fallback: use predict and create dummy probs
        y_pred = clf.predict(X_val)
        pos_proba = y_pred.astype(float)
    y_pred = clf.predict(X_val)
    print('\nEmbedding classifier report:')
    report = classification_report(y_val, y_pred, target_names=class_names)
    print(report)
    try:
        with open(out_dir / 'classification_report.txt', 'w', encoding='utf8') as f:
            f.write(report)
    except Exception:
        pass

    cm = confusion_matrix(y_val, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=list(range(len(class_names))), yticks=list(range(len(class_names))), xticklabels=class_names, yticklabels=class_names, ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    cm_path = out_dir / 'confusion_embed.png'
    fig.savefig(str(cm_path))
    print('Saved confusion matrix to', cm_path)

    # ROC and PR curves
    try:
        from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
        # compute ROC AUC
        auc = roc_auc_score(y_val, pos_proba)
        fpr, tpr, _ = roc_curve(y_val, pos_proba)
        prec, rec, _ = precision_recall_curve(y_val, pos_proba)
        ap = average_precision_score(y_val, pos_proba)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(fpr, tpr, label=f'AUC={auc:.4f}')
        ax[0].plot([0, 1], [0, 1], '--', color='gray')
        ax[0].set_title('ROC Curve')
        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')
        ax[0].legend()

        ax[1].plot(rec, prec, label=f'AP={ap:.4f}')
        ax[1].set_title('Precision-Recall')
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].legend()

        plt.tight_layout()
        roc_path = out_dir / 'roc_pr_curves.png'
        fig.savefig(str(roc_path))
        plt.close(fig)
        print('Saved ROC/PR curves to', roc_path)
    except Exception as e:
        print('Could not compute ROC/PR curves:', e)

    # save classifier and metadata
    try:
        import joblib as _joblib
        _joblib.dump({'model': clf, 'class_names': class_names}, str(out_dir / 'handwriting_embed_model.joblib'))
        with open(out_dir / 'metadata.json', 'w', encoding='utf8') as f:
            json.dump({'class_names': class_names, 'img_size': args.img_size, 'mode': args.mode}, f, indent=2)
        print('Saved embed model to', out_dir / 'handwriting_embed_model.joblib')
    except Exception as e:
        print('Failed to save model:', e)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    main(args)
