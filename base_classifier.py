import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        # Load the image with the target size required by MobileNetV2
        img = image.load_img(image_path, target_size=(224, 224))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Preprocess the image for the model
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions using the model
        predictions = model.predict(img_array)

        # Decode the predictions to get human-readable labels
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Print the top-3 predictions
        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

    except Exception as e:
        print(f"Error processing image: {e}")

# Example Usage
if __name__ == "__main__":
    image_path = "basic_cat.jpg"  # Replace with the path to your image file
    classify_image(image_path)