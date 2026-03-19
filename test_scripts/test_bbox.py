from PIL import Image, ImageDraw

def draw_bbox(image_path, box_norm, label="Object"):
    """
    box_norm: list of [xmin, ymin, xmax, ymax] in 0-1000 scale
    """
    # Load the image
    img = Image.open(image_path)
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # Denormalize coordinates
    # Formula: (coordinate / 1024) * dimension
    xmin = (box_norm[0] / 1024) * width
    ymin = (box_norm[1] / 1024) * height
    xmax = (box_norm[2] / 1024) * width
    ymax = (box_norm[3] / 1024) * height

    # Draw the rectangle (outline only)
    # Using a 5-pixel width for visibility
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=5)

    # Add a simple text label
    draw.text((xmin, ymin - 15), label, fill="red")

    # Display the result
    img.show()
    # img.save("output_with_box.jpg") # Uncomment to save

if __name__ == "__main__":
    # Example normalized box from Qwen: [xmin, ymin, xmax, ymax]
    # This represents a box roughly in the center-left
    my_box = [514, 438, 760, 715]
    
    # Replace 'image.jpg' with your actual filename
    try:
        draw_bbox("test_images/blocks.webp", my_box, label="Detected Item")
    except FileNotFoundError:
        print("Error: Please provide a valid path to an image file.")