from PIL import Image, ImageFilter, ImageOps # type: ignore
import matplotlib.pyplot as plt # type: ignore

# ====================================
# === Display/save helper function ===
# ====================================
def show_and_save(img, title, filename):
    # Display image using matplotlib and save to disk
    plt.imshow(img, cmap='gray') # Grayscale colormap improves visibility for edge/emboss filters
    plt.axis('off')              # Hide axes
    plt.title(title)             # Title displayed above the image
    plt.savefig(filename)        # Save output image
    print(f"Processed image saved as '{filename}'.")

# ========================
# === REQUIRED FILTERS ===
# ========================

# Applies Gaussian blur to soften the image
def apply_blur_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128)) # Standardize size
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2)) # Apply blur

        show_and_save(img_blurred, "Gaussian Blur", "blurred_image.png")

    except Exception as e:
        print(f"Error applying blur filter: {e}")

# Applies edge detection to highlight outlines and object boundaries
def apply_edge_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES) # Detect edges

        show_and_save(img_edges, "Edge Detection", "edge_detected_image.png")
    
    except Exception as e:
        print(f"Error applying edge detection: {e}")

# Sharpens the image to enhance edges and fine details
def apply_sharpen_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_sharp = img_resized.filter(ImageFilter.SHARPEN) # Apply sharpening

        show_and_save(img_sharp, "Sharpen", "sharpened_image.png")

    except Exception as e:
        print(f"Error applying sharpen filter: {e}")

# Applies emboss effect to simulate 3D engraving-like texture
def apply_emboss_filter(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_emboss = img_resized.filter(ImageFilter.EMBOSS) # Apply embossing

        show_and_save(img_emboss, "Emboss", "embossed_image.png")

    except Exception as e:
        print(f"Error applying emboss filter: {e}")

# ======================
# === CUSTOM FILTERS ===
# ======================

# Applies posterize effect to reduce color levels and create a stylized look
def apply_posterize_filter(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_posterized = posterize(img) # Apply custom posterize logic
        show_and_save(img_posterized, "Posterize", "posterized_image.png")
    except Exception as e:
        print(f"Error applying posterize filter: {e}")

# Applies pixelation effect to simplify image using large pixel blocks
def apply_pixelate_filter(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        img_pixelated = pixelate(img) # Apply custom pixelation logic
        show_and_save(img_pixelated, "Pixelate", "pixelated_image.png")
    except Exception as e:
        print(f"Error applying pixelate filter: {e}")

# ======================================
# === CUSTOM FILTER HELPER FUNCTIONS ===
# ======================================

# Pixelate core helper
def pixelate(img, pixel_size=2):
    width, height = img.size
    new_width = max(1, width // pixel_size) # Reduce dimensions based on pixel size
    new_height = max(1, height // pixel_size)

    # Downscale image to create pixel effect
    img_small = img.resize((new_width, new_height), resample=Image.BILINEAR)
    # Upscale with nearest neighbor to preserve blocky pixels
    img_pixelated = img_small.resize((width, height), Image.NEAREST)

    return img_pixelated

# Posterize core helper
def posterize(img, bits=3):
    # Reduces each pixel channel to one of 4 values (0, 64, 128, 192)
    return Image.eval(img, lambda x: x // 64 * 64)


# ===========================
# === FINAL CUSTOM FILTER ===
# ===========================

# Combines posterization and pixelation into one stylized effect
def apply_posterize_then_pixelate(image_path):
    try:
        img = Image.open(image_path).resize((128, 128))
        # Apply posterize first, then pixelate the result
        img_combo = pixelate(posterize(img))

        show_and_save(img_combo, "Posterize + Pixelate", "posterized_pixelated_image.png")
    except Exception as e:
        print(f"Error applying posterize + pixelate filter: {e}")

# =========================
# === MAIN SCRIPT ENTRY ===
# =========================

if __name__ == "__main__":
    image_path = "elephant.jpg"

    # Apply each filter to the image
    apply_blur_filter(image_path)
    apply_edge_filter(image_path)
    apply_sharpen_filter(image_path)
    apply_emboss_filter(image_path)
    apply_posterize_filter(image_path)
    apply_pixelate_filter(image_path)
    apply_posterize_then_pixelate(image_path)