import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from data_preprocessing import test_generator  # Import the test data generator from your preprocessing file
from tensorflow.keras.models import load_model
# from keras.src.saving import legacy_sm_saving_lib

# Path to your saved model
model_path = r"C:\Users\acer\Desktop\FLML\models\mobilenetv2_flower_classifier.keras"

# Load the saved model
# model = tf.keras.models.load_model(model_path)
# model = legacy_sm_saving_lib.load_model(model_path)
model = load_model(model_path)
print(f"Model loaded from {model_path}")
model.save('mobilenetv2_flower_classifier.keras') 

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Predict classes on test data
test_generator.reset()
pred_probs = model.predict(test_generator)
pred_classes = np.argmax(pred_probs, axis=1)

# True classes
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print("Classification Report:\n", report)

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()