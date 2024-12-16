from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

def classify_image_with_blur(image_path):
    """Resize the image, apply Gaussian blur, and classify the image."""
    try:
        # Load the image using PIL
        img = Image.open(image_path)

        # Resize the image to 128x128 pixels
        img_resized = img.resize((128, 128))

        # Apply Gaussian blur to the image
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        # Display the processed image
        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig("blurred_image.png")
        print("Processed image saved as 'blurred_image.png'.")

        # Convert the blurred image to a numpy array
        img_array = np.array(img_blurred)

        # Resize the array to 224x224 (required by MobileNetV2)
        img_array = Image.fromarray(img_array).resize((224, 224))
        img_array = np.array(img_array)

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
    classify_image_with_blur(image_path)