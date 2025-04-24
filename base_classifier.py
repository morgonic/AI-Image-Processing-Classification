import tensorflow as tf # type: ignore
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


model = MobileNetV2(weights="imagenet")


# --- Occlusion Functions ---
def apply_black_box(img, x, y, box_size=50):
    img_copy = img.copy()
    img_copy[y:y+box_size, x:x+box_size] = 0
    return img_copy

def apply_blur_patch(img, x, y, box_size=50):
    img_copy = img.copy()
    patch = img[y:y+box_size, x:x+box_size]
    blurred = cv2.GaussianBlur(patch, (15, 15), 0)
    img_copy[y:y+box_size, x:x+box_size] = blurred
    return img_copy

def apply_noise_patch(img, x, y, box_size=50):
    img_copy = img.copy()
    noise = np.random.randint(0, 256, (box_size, box_size, 3), dtype=np.uint8)
    img_copy[y:y+box_size, x:x+box_size] = noise
    return img_copy


# --- Grad-CAM Heatmap Functions ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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

    # Compute channel-wise mean of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs with the mean gradients
    conv_outputs = conv_outputs[0]
    conv_outputs = conv_outputs * tf.reshape(pooled_grads, [1, 1, -1])
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) 
    max_val = tf.math.reduce_max(heatmap)
    heatmap /= tf.maximum(max_val, tf.keras.backend.epsilon())
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", label=None, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap_color * alpha + img
    cv2.imwrite(cam_path, np.uint8(superimposed_img))

    # Save Grad-CAM plot
    plt.imshow(cv2.cvtColor(superimposed_img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')

    title = f"Grad-CAM: {label}" if label else "Grad-CAM"
    plt.title(title)

    filename = f"gradcam_plot_{label}.png" if label else "gradcam_plot.png"
    plt.savefig(filename)

# --- Image Classification Function ---
def classify_image(image_path):
    """Classify an image and display the predictions."""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        
        img_array = image.img_to_array(img)
        
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("Top-3 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            print(f"{i + 1}: {label} ({score:.2f})")

        # Grad-CAM visualization
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

# --- Occlusion Wrapper Function ---
def occlude_and_classify(image_path, x, y, box_size=50, occlusion_type='black'):
    try:
        # Load and preprocess original image
        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, (224, 224))

        # Apply occlusion
        if occlusion_type == 'black':
            occluded_img = apply_black_box(original_img, x, y, box_size)
        elif occlusion_type == 'blur':
            occluded_img = apply_blur_patch(original_img, x, y, box_size)
        elif occlusion_type == 'noise':
            occluded_img = apply_noise_patch(original_img, x, y, box_size)
        else:
            raise ValueError(f"Unknown occlusion type: {occlusion_type}")

        # Save and optionally view the occluded image
        occluded_path = f"occluded_{occlusion_type}_x{x}_y{y}_s{box_size}.jpg"
        cv2.imwrite(occluded_path, occluded_img)

        # Convert to PIL and classify
        img_pil = image.array_to_img(occluded_img)
        img_array = image.img_to_array(img_pil)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]

        print(f"\nTop-3 Predictions (with {occlusion_type} patch at ({x},{y})): ")
        for i, (imagenet_id, label, score) in enumerate(decoded):
            print(f"{i + 1}: {label} ({score:.2f})")

    except Exception as e:
        print(f"Error during occlusion classification: {e}")


# --- Image Classification Function Call ---
if __name__ == "__main__":
    image_path = "elephant.jpg"  
    classify_image(image_path)

    # Try occlusion at position (60, 60) with a 50x50 patch
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='black')
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='blur')
    occlude_and_classify(image_path, x=60, y=60, box_size=50, occlusion_type='noise')