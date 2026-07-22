import os
import sys
import json
import numpy as np
import tensorflow as tf

# Use tf.keras submodules directly for better IDE resolution (matches train.py)
image = tf.keras.preprocessing.image
preprocess_input = tf.keras.applications.resnet50.preprocess_input

def predict_tumor(img_path, model_path='models/best_brain_tumor_model.h5'):
    """Loads a trained model and predicts the specific tumor type from an MRI."""
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        # Fallback to final model if best model doesn't exist
        model_path = 'models/final_brain_tumor_model.h5'
        if not os.path.exists(model_path):
            print("Error: Trained model not found. Please run training first.")
            return

    # Load class mapping
    mapping_path = 'models/class_indices.json'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            class_indices = json.load(f)
        # Invert the dictionary to get {index: name} and format names
        labels = {v: k.replace('_', ' ').title() for k, v in class_indices.items()}
    else:
        # Hardcoded fallback mapping
        labels = {0: "Glioma Tumor", 1: "Meningioma Tumor", 2: "No Tumor", 3: "Pituitary Tumor"}

    print(f"Loading model from {model_path}...")
    # Load model (compile=False since we're just doing inference)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"Analyzing image: {img_path}...")
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    result_label = labels.get(predicted_class_idx, "Unknown")
    
    print("\n" + "="*40)
    print(f"{'PREDICTION RESULT':^40}")
    print("="*40)
    print(f" Diagnosis:  {result_label}")
    print(f" Confidence: {confidence:.2%}")
    print("="*40)
    
    if result_label != "No Tumor":
        print("\n[!] WARNING: This is an AI-generated prediction.")
        print("    Please consult a medical professional for clinical diagnosis.")
    else:
        print("\nAll clear! No tumor features detected by the model.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_file = sys.argv[1]
        predict_tumor(img_file)
    else:
        print("Usage: python predict.py <path_to_image>")
        print("Example: python predict.py data/Testing/glioma_tumor/image(1).jpg")
