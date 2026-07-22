import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
# Use tf.keras submodules directly to help IDE resolution
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
img_to_array = tf.keras.preprocessing.image.img_to_array
load_img = tf.keras.preprocessing.image.load_img

def visualize_samples(data_dir='data', num_samples=3):
    """Displays sample images from yes/no folders."""
    classes = ['yes', 'no']
    plt.figure(figsize=(12, 8))
    
    for i, cls in enumerate(classes):
        path = os.path.join(data_dir, cls)
        if not os.path.exists(path):
            continue
            
        images = os.listdir(path)
        samples = random.sample(images, min(num_samples, len(images)))
        
        for j, img_name in enumerate(samples):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(2, num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.title(f"Class: {cls}\n{img_name}")
            plt.axis('off')
            
    plt.tight_layout()
    plt.show()

def visualize_augmentation(img_path):
    """Shows how data augmentation transforms a single image."""
    if not os.path.exists(img_path):
        print(f"File {img_path} not found for augmentation demo.")
        return
        
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    # We use the same generator settings as in train.py
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    i = 2
    for batch in datagen.flow(x, batch_size=1):
        plt.subplot(3, 3, i)
        plt.imshow(batch[0].astype('uint8'))
        plt.title(f"Augmented {i-1}")
        plt.axis('off')
        i += 1
        if i > 9:
            break
            
    plt.tight_layout()
    plt.suptitle("How Data Augmentation Creates 'New' Images", fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Check if data exists
    if os.path.exists('data/yes'):
        print("Visualizing samples...")
        # visualize_samples()
        
        # Try to find one image to show augmentation
        sample_img_dir = 'data/yes'
        imgs = os.listdir(sample_img_dir)
        if imgs:
            sample_path = os.path.join(sample_img_dir, imgs[0])
            print(f"Visualizing augmentation for {sample_path}...")
            # Note: This requires a display environment. In terminal, this prints a placeholder.
            print("(Note: Plots will open in a new window if running in a GUI environment like VS Code)")
            visualize_augmentation(sample_path)
    else:
        print("Please place your data in the 'data/' folder first!")
