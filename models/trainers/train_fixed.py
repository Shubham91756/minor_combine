"""
============================================================
  Fixed Training Script - Proper Fine-Tuning
  Works with original OR preprocessed dataset
============================================================
"""

import os
import sys
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════
#  PREPROCESSING LAYER (SERIALIZABLE)
# ══════════════════════════════════════════════════════════════════════

@keras.utils.register_keras_serializable()
class PreprocessInput(layers.Layer):
    def __init__(self, model_name, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def call(self, inputs):
        x = inputs * 255.0  # inputs are in [0,1]; scale to [0,255] for preprocess_input
        if self.model_name == "ResNet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
            return preprocess_input(x)
        if self.model_name == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            return preprocess_input(x)
        if self.model_name == "DenseNet121":
            from tensorflow.keras.applications.densenet import preprocess_input
            return preprocess_input(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION - CHANGE DATA_DIR AFTER PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

DATA_DIR = "Handwriting"  # Change to "Handwriting_Processed" after preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VAL_SPLIT = 0.20

# Training params
EPOCHS_FROZEN = 20
LR_FROZEN = 1e-3
EPOCHS_FINETUNE = 20
LR_FINETUNE = 2e-5  # Lower for fine-tuning, slightly higher for better convergence

EPOCHS_SCRATCH = 35
LR_SCRATCH = 1e-4

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
LABEL_MAP = {"healthy": 0, "patient": 1}
MODEL_SAVE_PATH = "parkinson_model_best.h5"
USE_CLASS_WEIGHTS = False  # Set True if class imbalance is hurting validation


# ══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def collect_images(root):
    samples = []
    for class_name, label in LABEL_MAP.items():
        class_dir = Path(root) / class_name
        if not class_dir.is_dir():
            continue
        for file_path in class_dir.rglob("*"):
            if "__MACOSX" in file_path.parts:
                continue
            if file_path.is_file() and file_path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((str(file_path), label))
    return samples


def load_dataset(samples):
    images, labels = [], []
    for idx, (path, label) in enumerate(samples):
        try:
            img = load_img(path, target_size=IMG_SIZE, color_mode="rgb")
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
        except:
            continue
        if (idx + 1) % 200 == 0:
            print(f"    {idx + 1}/{len(samples)}", end='\r')
    print(f"    {len(images)}/{len(samples)} loaded    ")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════
#  AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

def build_augmentation():
    return keras.Sequential([
        layers.RandomRotation(0.08, seed=SEED),
        layers.RandomZoom(0.10, seed=SEED),
        layers.RandomTranslation(0.05, 0.05, seed=SEED),
        layers.RandomContrast(0.10, seed=SEED),
    ], name="augmentation")


# ══════════════════════════════════════════════════════════════════════
#  MODEL BUILDING
# ══════════════════════════════════════════════════════════════════════

def try_load_pretrained(name, input_shape):
    print(f"    {name}...", end=" ")
    try:
        if name == "ResNet50":
            from tensorflow.keras.applications import ResNet50
            base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
            finetune_from = 140
        elif name == "MobileNetV2":
            from tensorflow.keras.applications import MobileNetV2
            base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
            finetune_from = 100
        elif name == "DenseNet121":
            from tensorflow.keras.applications import DenseNet121
            base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
            finetune_from = 310
        else:
            return None, False, None
        
        print("✓")
        return base, True, finetune_from
    except Exception as e:
        print(f"✗")
        return None, False, None


def build_custom_cnn(input_shape):
    return keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name="custom_cnn")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("\n" + "=" * 70)
    print("  PARKINSON'S DETECTION TRAINING")
    print("=" * 70 + "\n")

    # Load data
    print("[1/4] Loading dataset...")
    samples = collect_images(DATA_DIR)
    if not samples:
        print(f"[ERROR] No images in {DATA_DIR}")
        sys.exit(1)
    
    healthy = sum(1 for _, l in samples if l == 0)
    patient = sum(1 for _, l in samples if l == 1)
    print(f"  Found: {len(samples)} (healthy={healthy}, patient={patient})\n")

    print("  Loading images...")
    X, y = load_dataset(samples)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=SEED
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]}\n")
    
    class_weight = None
    if USE_CLASS_WEIGHTS:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight = {i: w for i, w in enumerate(class_weights)}
        print(f"  Using class weights: {class_weight}\n")

    # Build model
    print("[2/4] Building model...")
    print("  Trying pretrained models:")
    
    base_model = None
    finetune_from = None
    is_pretrained = False
    selected_model_name = None
    
    for model_name in ["ResNet50", "MobileNetV2", "DenseNet121"]:
        base_model, success, finetune_from = try_load_pretrained(model_name, (*IMG_SIZE, 3))
        if success:
            print(f"\n  Using {model_name}\n")
            is_pretrained = True
            selected_model_name = model_name
            break
    
    if not is_pretrained:
        print("\n  No pretrained model available - building custom CNN\n")
        base_model = build_custom_cnn((*IMG_SIZE, 3))
    else:
        print(f"  Preprocessing: {selected_model_name} preprocess_input\n")
    
    # Build full model with augmentation
    aug = build_augmentation()
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = aug(inputs)
    if is_pretrained and selected_model_name:
        x = PreprocessInput(selected_model_name, name="preprocess")(x)
    
    if is_pretrained:
        base_model.trainable = False  # Start frozen
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = base_model(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(f"  Total params: {model.count_params():,}\n")

    # Training
    print("[3/4] Training...\n")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]

    if is_pretrained:
        # Phase 1: Train head
        print("  PHASE 1: Training head (base frozen)")
        print("  " + "-" * 66)
        
        model.compile(
            optimizer=keras.optimizers.Adam(LR_FROZEN),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train, y_train,
            epochs=EPOCHS_FROZEN,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        # Phase 2: Fine-tune - FIXED VERSION
        print(f"\n  PHASE 2: Fine-tuning (unfreezing from layer {finetune_from})")
        print("  " + "-" * 66)
        
        # Properly unfreeze base model
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        for i, layer in enumerate(base_model.layers):
            if i < finetune_from:
                layer.trainable = False
            else:
                layer.trainable = True
        
        # Count trainable params to verify
        trainable_count = sum([keras.backend.count_params(w) for w in model.trainable_weights])
        print(f"  Trainable parameters: {trainable_count:,}")
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(LR_FINETUNE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train, y_train,
            epochs=EPOCHS_FINETUNE,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
    else:
        # Train from scratch
        print("  Training from scratch")
        print("  " + "-" * 66)
        
        model.compile(
            optimizer=keras.optimizers.Adam(LR_SCRATCH),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.fit(
            X_train, y_train,
            epochs=EPOCHS_SCRATCH,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )

    # Evaluation
    print("\n[4/4] Final Evaluation")
    print("=" * 70)
    
    best_model = keras.models.load_model(MODEL_SAVE_PATH)
    
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n  Training   → Acc: {train_acc:.4f} | Loss: {train_loss:.4f}")
    print(f"  Validation → Acc: {val_acc:.4f} | Loss: {val_loss:.4f}")
    
    print(f"\n  ✓ Best model: {MODEL_SAVE_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
