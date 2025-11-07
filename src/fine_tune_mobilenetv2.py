import os
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import train_generator, validation_generator  # Import your data generators

IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 5  # Change to your number of flower classes
EPOCHS = 10
INITIAL_EPOCHS = 15  # Number of epochs your original model was trained

# Load the base MobileNetV2 model pretrained on ImageNet without the top layers
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze all layers initially
base_model.trainable = False

# Build the classification head on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile and train the model initially (if not done yet)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Load weights from your previously trained model
model.load_weights(r'C:\Users\zackb\OneDrive\Desktop\Flower_Classification\models\mobilenetv2_flower_classifier.h5')
print("Loaded trained weights")

# Unfreeze some layers from the base model for fine-tuning
base_model.trainable = True

# Fine-tune from this layer onwards (can adjust)
fine_tune_at = 100  # unfreeze from this layer to end
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with fine-tuning
fine_tune_epochs = 10
total_epochs = INITIAL_EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=total_epochs,
    initial_epoch=INITIAL_EPOCHS
)

# Save the fine-tuned model
os.makedirs("models", exist_ok=True)
model.save('models/mobilenetv2_flower_classifier_finetuned.h5')
print("Fine-tuned model saved!")
