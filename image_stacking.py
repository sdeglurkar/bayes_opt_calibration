from PIL import Image

# List your PNGs in order
images = ["gp_1_0.png", "gp_2_0.png", "gp_5_0.png", "gp_10_0.png", "gp_25_0.png", "gp_39_0.png"]  # first image has white background

def make_white_transparent(img):
    """Convert white pixels to transparent."""
    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    for item in datas:
        # If pixel is nearly white, make it transparent
        if item[0] > 250 and item[1] > 250 and item[2] > 250:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img

# Open the first image as base (keep white background)
base = Image.open(images[0]).convert("RGBA")

# Process remaining images: make white transparent, then overlay
for img_path in images[1:]:
    overlay = Image.open(img_path)
    overlay = make_white_transparent(overlay)
    # Ensure same size as base
    if overlay.size != base.size:
        overlay = overlay.resize(base.size)
    base = Image.alpha_composite(base, overlay)

# Save final combined image
base.save("combined.png")
print("Combined image saved as combined.png")