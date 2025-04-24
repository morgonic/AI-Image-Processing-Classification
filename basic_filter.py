from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Display/save function
def show_and_save(img, title, filename):
    plt.imshow(img, cmap='gray') # Using grayscale colormap for better edge/emboss visibility
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    print(f"Processed image saved as '{filename}'.")

# Guassian Blur
def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        show_and_save(img_blurred, "Gaussian Blur", "blurred_image.png")

    except Exception as e:
        print(f"Error applying blur filter: {e}")

# Edge Detection
def apply_edge_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)

        show_and_save(img_edges, "Edge Detection", "edge_detected_image.png")
    
    except Exception as e:
        print(f"Error applying edge detection: {e}")

# Sharpen
def apply_sharpen_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_sharp = img_resized.filter(ImageFilter.SHARPEN)

        show_and_save(img_sharp, "Sharpen", "sharpened_image.png")

    except Exception as e:
        print(f"Error applying sharpen filter: {e}")

# Emboss
def apply_emboss_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_emboss = img_resized.filter(ImageFilter.EMBOSS)

        show_and_save(img_emboss, "Emboss", "embossed_image.png")

    except Exception as e:
        print(f"Error applying emboss filter: {e}")
    
# Run them all
if __name__ == "__main__":
    image_path = "elephant.jpg"

    apply_blur_filter(image_path)
    apply_edge_filter(image_path)
    apply_sharpen_filter(image_path)
    apply_emboss_filter(image_path)