import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the saved fine-tuned model
MODEL_PATH = r'C:\Users\acer\Desktop\FLML\models\mobilenetv2_flower_classifier.keras'

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# Assuming class indices from your training data preprocessing
class_indices = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4
}
# Reverse mapping from class index to class label
class_labels = {v: k for k, v in class_indices.items()}

def preprocess_image(img_path):
    """Load and preprocess image."""
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(img_path):
    """Predict flower class for a single image."""
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]
    label = class_labels[predicted_class]
    return label, confidence

def display_image(img_path, label, confidence):
    """Display image with predicted label and confidence."""
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.title(f"Predicted: {label} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Example image path to test
    test_image_path = r'C:\Users\acer\Desktop\FLML\brose.jpeg'  # Change to your image path

    label, confidence = predict_image(test_image_path)
    print(f"Prediction: {label} with confidence {confidence*100:.2f}%")
    display_image(test_image_path, label, confidence)
