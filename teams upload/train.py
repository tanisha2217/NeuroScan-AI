import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Use tf.keras submodules directly to help IDE resolution
layers = tf.keras.layers
models = tf.keras.models
callbacks = tf.keras.callbacks
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ResNet50 = tf.keras.applications.resnet50.ResNet50
preprocess_input = tf.keras.applications.resnet50.preprocess_input

# ============================================================
# 1. Configuration
# ============================================================
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 64      # Larger batch = fewer steps per epoch = faster
TRAIN_DIR   = os.path.join('data', 'Training')
TEST_DIR    = os.path.join('data', 'Testing')
NUM_CLASSES = 4
EPOCHS      = 20      # Fine-tuning phase max epochs (EarlyStopping kicks in earlier)

# ============================================================
# 2. Speed: Enable Mixed Precision (faster on modern CPUs too)
# ============================================================
tf.keras.mixed_precision.set_global_policy('float32')  # keep float32 for CPU stability

# ============================================================
# 3. Model: ResNet50 + Custom Head (4-class softmax)
# ============================================================
def build_model():
    """Builds a ResNet50-based transfer learning model for 4-class classification."""
    print("Building 4-class ResNet50 model...")

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False   # Freeze base initially

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    # Label smoothing helps prevent the model from being overconfident on hard classes
    predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        # Label smoothing (0.1) prevents overconfidence on hard-to-distinguish classes
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    print(f"Model built. Output: {NUM_CLASSES} classes (softmax).")    
    return model, base_model

# ============================================================
# 3. Data Generators
# ============================================================
def get_data_generators():
    """Sets up image data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Save class indices mapping so backend can decode predictions
    os.makedirs('models', exist_ok=True)
    with open('models/class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)
    print(f"Class indices saved: {train_gen.class_indices}")

    # Compute class weights
    labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights computed: {class_weights}")

    return train_gen, val_gen, class_weights

# ============================================================
# 4. Training Pipeline
# ============================================================
def train():
    """Main training pipeline with fine-tuning."""
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory '{TRAIN_DIR}' not found.")
        print("Please ensure data/Training/ and data/Testing/ folders exist.")
        return

    train_gen, val_gen, class_weights = get_data_generators()
    model, base_model = build_model()

    # ---- Phase 1: Train only the custom head ----
    print("\n--- Phase 1: Training custom head (base frozen) ---")
    phase1_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint('models/best_brain_tumor_model.h5', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    ]

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=phase1_callbacks
    )

    # ---- Phase 2: Fine-tune top layers of ResNet50 ----
    print("\n--- Phase 2: Fine-tuning top ResNet50 layers ---")
    # Unfreeze the last 50 layers for deeper fine-tuning
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    phase2_callbacks = [
        callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        callbacks.ModelCheckpoint('models/best_brain_tumor_model.h5', save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-8, verbose=1),
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=phase2_callbacks
    )

    # ---- Save everything ----
    model.save('models/final_brain_tumor_model.h5')
    print("\nModel saved.")

    # Combine histories
    combined = {}
    for k in history1.history:
        combined[k] = history1.history[k] + history2.history[k]
    with open('models/training_history.json', 'w') as f:
        json.dump(combined, f)
    print("Training history saved.")

    # ---- Evaluation ----
    print("\n--- Final Evaluation on Test Set ---")
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = val_gen.classes

    class_names = list(val_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("Confusion matrix saved to models/confusion_matrix.png")

    return model

if __name__ == "__main__":
    train()
