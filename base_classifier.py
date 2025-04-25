import tensorflow as tf # type: ignore
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
import warnings

# Ignore UserWarnings from TensorFlow/Keras
warnings.filterwarnings("ignore", category=UserWarning)

#Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights="imagenet")

# ===========================
# === Occlusion Functions ===
# ===========================

# Apply a black square occlusion patch
def apply_black_box(img, x, y, box_size=50):
    img_copy = img.copy()
    img_copy[y:y+box_size, x:x+box_size] = 0
    return img_copy

# Apply Gaussian blur to a square patch
def apply_blur_patch(img, x, y, box_size=50):
    img_copy = img.copy()
    patch = img[y:y+box_size, x:x+box_size]
    blurred = cv2.GaussianBlur(patch, (15, 15), 0)
    img_copy[y:y+box_size, x:x+box_size] = blurred
    return img_copy

# Apply random noise to a square patch
def apply_noise_patch(img, x, y, box_size=50):
    img_copy = img.copy()
    noise = np.random.randint(0, 256, (box_size, box_size, 3), dtype=np.uint8)
    img_copy[y:y+box_size, x:x+box_size] = noise
    return img_copy

# ==================================
# === Grad-CAM Heatmap Functions ===
# ==================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that maps input image to activations of the target layer and final output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Get gradients of the top predicted class
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None. Check if the target layer is connected to the output.")

    # Average gradients over width and height to get importance for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by its importance
    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs * tf.reshape(pooled_grads, [1, 1, -1])

    # Sum along the channels to get a heatmap
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) 
    max_val = tf.math.reduce_max(heatmap)
    heatmap /= tf.maximum(max_val, tf.keras.backend.epsilon())
    
    return heatmap.numpy()

# Overlay the Grad-CAM heatmap onto the original image and save it
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", label=None, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # Convert to 8-bit
    heatmap = np.uint8(255 * heatmap)
    # Apply color
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap onto image
    superimposed_img = heatmap_color * alpha + img
    # Save result
    cv2.imwrite(cam_path, np.uint8(superimposed_img))

    # Create and save Grad-CAM plot
    plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    title = f"Grad-CAM: {label}" if label else "Grad-CAM"
    plt.title(title)

    filename = f"gradcam_plot_{label}.png" if label else "gradcam_plot.png"
    plt.savefig(filename)

# =====================================
# === Image Classification Function ===
# =====================================

def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using MobileNetV2
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Print top-3 results
        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Generate Grad-CAM for the top prediction
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1")
        top_label = decoded_predictions[0][1]
        top_score = decoded_predictions[0][2]
        label_text = f"{top_label} ({top_score:.2f})"
        save_and_display_gradcam(
            image_path, 
            heatmap, 
            cam_path=f"gradcam_{top_label}.jpg",
            label=label_text
        )

    except Exception as e:
        print(f"Error processing image: {e}")

# ==================================
# === Occlusion Wrapper Function ===
# ==================================

def occlude_and_classify(image_path, x, y, box_size=50, occlusion_type='black'):
    """Apply an occlusion to a specific region of the image, classify, and display predictions."""
    try:
        # Load and resize original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))

        # Apply chosen occlusion method
        if occlusion_type == 'black':
            occluded_img = apply_black_box(original_img, x, y, box_size)
        elif occlusion_type == 'blur':
            occluded_img = apply_blur_patch(original_img, x, y, box_size)
        elif occlusion_type == 'noise':
            occluded_img = apply_noise_patch(original_img, x, y, box_size)
        else:
            raise ValueError(f"Unknown occlusion type: {occlusion_type}")

        # Save occluded image
        occluded_path = f"occluded_{occlusion_type}_x{x}_y{y}_s{box_size}.jpg"
        cv2.imwrite(occluded_path, occluded_img)

        # Convert occluded image to array for classification
        img_pil = image.array_to_img(occluded_img)
        img_array = image.img_to_array(img_pil)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # Run prediction on occluded image
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]

        # Print prediction results
        print(f"\nTop-3 Predictions (with {occlusion_type} patch at ({x},{y})): ")
        for i, (imagenet_id, label, score) in enumerate(decoded):
            print(f"{i + 1}: {label} ({score:.2f})")

    except Exception as e:
        print(f"Error during occlusion classification: {e}")

# ===============================
# === Main Script Entry Point ===
# ===============================

if __name__ == "__main__":
    # Replace with your own image file
    image_path = "elephant.jpg"  
    # Run standard classification and Grad-CAM
    classify_image(image_path)

    # Apply occlusion patches and classify
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='black')
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='blur')
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='noise')