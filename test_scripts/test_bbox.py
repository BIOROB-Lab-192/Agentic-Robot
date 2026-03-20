import cv2

# Assume 'img' is your loaded image (numpy array)
# img.shape returns (height, width, channels), e.g., (1080, 1920, 3)

img = cv2.imread("test_images/dog.jpg", -1)

height, width = img.shape[:2]

# My output: [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
box_normalized = [0.402, 0.426, 0.673, 0.715]

# Step 1: Convert normalized to pixels
x_min_px = int(box_normalized[0] * width)
y_min_px = int(box_normalized[1] * height)
x_max_px = int(box_normalized[2] * width)
y_max_px = int(box_normalized[3] * height)

# Step 2: Define the bounding box
# Note: OpenCV expects [x, y, w, h] usually for drawing, or [x_min, y_min, x_max, y_max] for cv2.rectangle
x, y, w, h = x_min_px, y_min_px, (x_max_px - x_min_px), (y_max_px - y_min_px)

# Step 3: Draw it on the image to verify
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Optional: Crop and save the region of interest
roi = img[y : y + h, x : x + w]
cv2.imwrite("dog_crop.jpg", roi)
