from PIL import Image, ImageFilter, ImageOps # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Display/save helper function
def show_and_save(img, title, filename):
    plt.imshow(img, cmap='gray') # Using grayscale colormap for better edge/emboss visibility
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)
    print(f"Processed image saved as '{filename}'.")

# --- REQUIRED FILTERS ---

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

# --- CUSTOM FILTERS ---

# Posterize
def apply_posterize_filter(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_posterized = posterize(img)
        show_and_save(img_posterized, "Posterize", "posterized_image.png")
    except Exception as e:
        print(f"Error applying posterize filter: {e}")

# Pixelate
def apply_pixelate_filter(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_pixelated = pixelate(img)
        show_and_save(img_pixelated, "Pixelate", "pixelated_image.png")
    except Exception as e:
        print(f"Error applying pixelate filter: {e}")

# --- CUSTOM FILTER HELPER FUNCTIONS ---

# Pixelate core helper
def pixelate(img, pixel_size=2):
    width, height = img.size
    new_width = max(1, width // pixel_size)
    new_height = max(1, height // pixel_size)

    img_small = img.resize((new_width, new_height), resample=Image.BILINEAR)
    img_pixelated = img_small.resize((width, height), Image.NEAREST)

    return img_pixelated

# Posterize core helper
def posterize(img, bits=3):
    return Image.eval(img, lambda x: x // 64 * 64)


# --- FINAL CUSTOM FILTER ---

# Posterize then Pixelate
def apply_posterize_then_pixelate(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_combo = pixelate(posterize(img))
        show_and_save(img_combo, "Posterize + Pixelate", "posterized_pixelated_image.png")
    except Exception as e:
        print(f"Error applying posterize + pixelate filter: {e}")
    
    
# Run them all
if __name__ == "__main__":
    image_path = "elephant.jpg"

    apply_blur_filter(image_path)
    apply_edge_filter(image_path)
    apply_sharpen_filter(image_path)
    apply_emboss_filter(image_path)
    apply_posterize_filter(image_path)
    apply_pixelate_filter(image_path)
    apply_posterize_then_pixelate(image_path)