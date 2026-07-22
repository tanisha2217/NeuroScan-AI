import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from io import BytesIO
from PIL import Image
import base64
import cv2
import matplotlib.cm as cm

app = Flask(__name__)
CORS(app)

# ============================================================
# Configuration
# ============================================================
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, '..', 'models', 'best_brain_tumor_model.h5')
FINAL_MODEL_PATH  = os.path.join(BASE_DIR, '..', 'models', 'final_brain_tumor_model.h5')
CLASS_INDEX_PATH  = os.path.join(BASE_DIR, '..', 'models', 'class_indices.json')

# Human-readable labels with clinical info
CLASS_INFO = {
    "glioma_tumor": {
        "label": "Glioma Tumor",
        "color": "red",
        "description": "Gliomas arise from glial cells. They are the most common type of malignant brain tumor.",
    },
    "meningioma_tumor": {
        "label": "Meningioma Tumor",
        "color": "orange",
        "description": "Meningiomas arise from the meninges. Usually benign and slow-growing.",
    },
    "no_tumor": {
        "label": "No Tumor Detected",
        "color": "teal",
        "description": "No significant neoplastic mass detected in the provided scan.",
    },
    "pituitary_tumor": {
        "label": "Pituitary Tumor",
        "color": "purple",
        "description": "Pituitary tumors form in the pituitary gland at the base of the brain.",
    },
}

# Global state
model        = None
class_names  = None   # index → class key, e.g. {0: 'glioma_tumor', ...}

# ============================================================
# Model Loading
# ============================================================
def load_brain_model():
    global model, class_names

    # Load class index mapping
    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH) as f:
            raw = json.load(f)
        # raw = {"glioma_tumor": 0, ...} → invert to {0: "glioma_tumor"}
        class_names = {v: k for k, v in raw.items()}
        print(f"Class names loaded: {class_names}")
    else:
        print("WARNING: class_indices.json not found. Using default order.")
        class_names = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}

    path = MODEL_PATH if os.path.exists(MODEL_PATH) else FINAL_MODEL_PATH
    print(f"Checking for model at: {os.path.abspath(path)}")

    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print("Model loaded successfully!")
            print(f"Output shape: {model.output_shape}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"ERROR: Model not found at {os.path.abspath(path)}")

# ============================================================
# Grad-CAM  (FIXED: uses class-specific gradient)
# ============================================================
def get_gradcam_heatmap(img_array, model, pred_class_idx, last_conv_layer="conv5_block3_out"):
    """
    Generates a Grad-CAM heatmap for the PREDICTED class.
    Key fix: gradient is computed w.r.t. preds[:, pred_class_idx],
    not always preds[:, 0], so the heatmap highlights the correct region.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        # Use the PREDICTED class neuron — this is the critical fix
        class_channel = preds[:, pred_class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap = heatmap / max_val
    return heatmap.numpy()

# ============================================================
# Apply heatmap overlay
# ============================================================
def apply_heatmap(img_buf, heatmap, alpha=0.45):
    img = Image.open(img_buf).convert('RGB')
    orig_size = img.size          # (W, H)
    img_np   = np.array(img)

    # Resize heatmap to original image size
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, orig_size)

    jet        = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]           # (H, W, 3) float 0-1
    jet_heatmap = (jet_heatmap * 255).astype(np.uint8)  # uint8

    # Blend
    superimposed = (jet_heatmap * alpha + img_np * (1 - alpha)).astype(np.uint8)
    result_img   = Image.fromarray(superimposed)

    buf = BytesIO()
    result_img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# ============================================================
# Localization & Bounding Box
# ============================================================
def get_localization_text(heatmap):
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    ny   = y / heatmap.shape[0]
    nx   = x / heatmap.shape[1]

    if 0.35 <= nx <= 0.65 and 0.35 <= ny <= 0.65:
        return "Central Brain Region"

    parts = []
    parts.append("Superior" if ny < 0.5 else "Inferior")
    parts.append("Left Hemispheric" if nx < 0.5 else "Right Hemispheric")
    return " ".join(parts) + " Region"

def get_tumor_bbox(heatmap, threshold=0.50):
    if heatmap is None or np.max(heatmap) == 0:
        return None

    # Smooth and threshold
    blurred = cv2.GaussianBlur(heatmap.astype(np.float32), (9, 9), 0)
    _, thresh = cv2.threshold(
        np.uint8(255 * blurred),
        int(255 * threshold),
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Estimated diameter  (224 px ≈ 22 cm field of view)
    diameter_cm = round((np.sqrt(w**2 + h**2) / 224) * 22, 1)

    H, W = heatmap.shape
    return {
        "x":        round(x / W * 100, 2),
        "y":        round(y / H * 100, 2),
        "w":        round(w / W * 100, 2),
        "h":        round(h / H * 100, 2),
        "cx":       round((x + w / 2) / W * 100, 2),
        "cy":       round((y + h / 2) / H * 100, 2),
        "diameter": diameter_cm,
    }

# ============================================================
# Routes
# ============================================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ready" if model else "model_missing"})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        load_brain_model()
        if not model:
            return jsonify({"error": "Model not loaded. Please run python src/train.py first."}), 500

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    raw_bytes = file.read()   # read once, reuse via BytesIO

    try:
        # Preprocess
        img = Image.open(BytesIO(raw_bytes)).convert('RGB').resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        # Predict
        preds          = model.predict(img_array, verbose=0)    # shape (1, 4)
        pred_idx       = int(np.argmax(preds[0]))
        confidence     = float(np.max(preds[0]))
        class_key      = class_names.get(pred_idx, 'unknown')
        info           = CLASS_INFO.get(class_key, {"label": class_key, "color": "gray", "description": ""})
        has_tumor      = class_key != "no_tumor"

        # ---- Grad-CAM + localization (only when tumor present) ----
        heatmap_b64    = None
        localization   = "N/A"
        bbox           = None

        if has_tumor:
            try:
                heatmap      = get_gradcam_heatmap(img_array, model, pred_idx)
                localization = get_localization_text(heatmap)
                bbox         = get_tumor_bbox(heatmap)
                heatmap_b64  = apply_heatmap(BytesIO(raw_bytes), heatmap)
            except Exception as e:
                print(f"Grad-CAM error: {e}")

        # Build per-class probability breakdown
        prob_breakdown = {
            class_names.get(i, f"class_{i}"): round(float(preds[0][i]) * 100, 2)
            for i in range(len(preds[0]))
        }

        return jsonify({
            "diagnosis":    info["label"],
            "tumor_type":   class_key,
            "has_tumor":    has_tumor,
            "confidence":   round(confidence * 100, 2),
            "color":        info["color"],
            "description":  info["description"],
            "localization": localization,
            "heatmap":      heatmap_b64,
            "bbox":         bbox,
            "breakdown":    prob_breakdown,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_brain_model()
    app.run(debug=True, port=5000)
