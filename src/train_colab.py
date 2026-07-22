# ============================================================
# NeuroScan AI — Google Colab Training Script
# Copy each cell block into a new Colab cell
# Runtime → Change runtime type → T4 GPU  (do this FIRST)
# ============================================================

# ── CELL 1: Check GPU ──────────────────────────────────────
import tensorflow as tf
print("TF Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ── CELL 2: Mount Google Drive ─────────────────────────────
from google.colab import drive  # type: ignore  (Colab-only module, not available locally)
drive.mount('/content/drive')

# ── CELL 3: Point to your dataset ──────────────────────────
# IMPORTANT: Upload the Training/ and Testing/ folders from the
# Kaggle dataset to your Google Drive first.
# Place them at: My Drive/NeuroScan/Training/ and My Drive/NeuroScan/Testing/
import os

DRIVE_BASE  = '/content/drive/MyDrive/NeuroScan'
TRAIN_DIR   = f'{DRIVE_BASE}/Training'
TEST_DIR    = f'{DRIVE_BASE}/Testing'
MODEL_SAVE  = f'{DRIVE_BASE}/models'
os.makedirs(MODEL_SAVE, exist_ok=True)

# Verify folders exist
for d in [TRAIN_DIR, TEST_DIR]:
    classes = os.listdir(d)
    counts  = {c: len(os.listdir(os.path.join(d, c))) for c in classes}
    print(f"\n{d}:")
    for c, n in counts.items():
        print(f"  {c}: {n} images")

# ── CELL 4: Enable Mixed Precision (2x speed on GPU) ───────
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Compute policy:", tf.keras.mixed_precision.global_policy())

# ── CELL 5: Configuration ──────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 64          # GPU can handle larger batches
NUM_CLASSES = 4
EPOCHS_P1   = 10          # Phase 1: head only
EPOCHS_P2   = 20          # Phase 2: fine-tuning

layers          = tf.keras.layers
models          = tf.keras.models
callbacks_api   = tf.keras.callbacks
ImageDataGen    = tf.keras.preprocessing.image.ImageDataGenerator
ResNet50        = tf.keras.applications.resnet50.ResNet50
preprocess      = tf.keras.applications.resnet50.preprocess_input

# ── CELL 6: Build Model ────────────────────────────────────
def build_model():
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    # Cast back to float32 for mixed-precision stability
    x = layers.Activation('linear', dtype='float32')(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    print("Model built:", model.output_shape)
    return model, base

model, base_model = build_model()

# ── CELL 7: Data Generators ────────────────────────────────
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

train_datagen = ImageDataGen(
    preprocessing_function=preprocess,
    rotation_range=25,
    width_shift_range=0.15, height_shift_range=0.15,
    shear_range=0.15, zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)
val_datagen = ImageDataGen(preprocessing_function=preprocess)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# Save class indices to Drive
with open(f'{MODEL_SAVE}/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
print("Class indices:", train_gen.class_indices)

# Compute class weights
labels = train_gen.classes
cw_array = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(cw_array))
print("Class weights:", class_weights)

# ── CELL 8: Phase 1 Training — Head Only ───────────────────
BEST_MODEL_PATH = f'{MODEL_SAVE}/best_brain_tumor_model.h5'

cbs_p1 = [
    callbacks_api.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    callbacks_api.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, verbose=1),
    callbacks_api.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
]

print("\n=== Phase 1: Training head (base frozen) ===")
history1 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_P1, class_weight=class_weights,
    callbacks=cbs_p1
)

# ── CELL 9: Phase 2 Training — Fine-Tune ResNet ────────────
# Unfreeze last 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

cbs_p2 = [
    callbacks_api.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    callbacks_api.ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True, verbose=1),
    callbacks_api.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1),
]

print("\n=== Phase 2: Fine-tuning top 50 ResNet layers ===")
history2 = model.fit(
    train_gen, validation_data=val_gen,
    epochs=EPOCHS_P2, class_weight=class_weights,
    callbacks=cbs_p2
)

# ── CELL 10: Save Final Model & History ────────────────────
model.save(f'{MODEL_SAVE}/final_brain_tumor_model.h5')
print("Models saved to Google Drive!")

combined = {}
for k in history1.history:
    combined[k] = history1.history.get(k, []) + history2.history.get(k, [])
with open(f'{MODEL_SAVE}/training_history.json', 'w') as f:
    json.dump(combined, f)
print("Training history saved.")

# ── CELL 11: Evaluation ────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

val_gen.reset()
preds       = model.predict(val_gen, verbose=1)
pred_cls    = np.argmax(preds, axis=1)
true_cls    = val_gen.classes
class_names = list(val_gen.class_indices.keys())

print("\n=== Classification Report ===")
print(classification_report(true_cls, pred_cls, target_names=class_names))

cm = confusion_matrix(true_cls, pred_cls)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{MODEL_SAVE}/confusion_matrix.png')
plt.show()
print("Confusion matrix saved to Google Drive!")

# ── CELL 12: Download model files to your PC ───────────────
# Run this after training to download the model files directly
from google.colab import files  # type: ignore  (Colab-only module, not available locally)
files.download(BEST_MODEL_PATH)
files.download(f'{MODEL_SAVE}/class_indices.json')
files.download(f'{MODEL_SAVE}/training_history.json')
print("Download started! Save these to your local: models/ folder")
