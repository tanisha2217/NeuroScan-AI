# Brain Tumor Detection using Deep Learning (ResNet50)

This project uses a pre-trained **ResNet50** model and **Transfer Learning** to detect brain tumors from MRI/CT scan images.

## Features
- **Transfer Learning**: Uses ResNet50 for high-accuracy feature extraction.
- **Data Augmentation**: Handles small datasets using random rotations, flips, and zooms.
- **Easy Inference**: Simple command-line script to test new images.
- **Learning Focused**: well-commented code for students learning TensorFlow/Keras.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
Download the [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) from Kaggle.

Structure the `data/` folder as follows:
```
data/
├── yes/  (Images WITH tumor)
└── no/   (Images WITHOUT tumor)
```

### 3. Verify Data
Run the setup script to ensure your folders are correct:
```bash
python src/data_setup.py
```

### 4. Train the Model
Start the training process:
```bash
python src/train.py
```
The best model will be saved in `models/best_brain_tumor_model.h5`.

### 5. Run Prediction (Inference)
Test the model on a single brain image:
```bash
python predict.py "path/to/your/image.jpg"
```

## How it Works
1. **Preprocessing**: Images are resized to 224x224 and normalized using ResNet's specific requirements.
2. **Feature Extraction**: The ResNet50 base identifies complex patterns in the brain tissue.
3. **Classification**: A dense layer with Sigmoid activation outputs the probability of a tumor.

## Disclaimer
This project is for educational and research purposes only. It is not intended for clinical use. Always consult with a medical professional for diagnosis.
