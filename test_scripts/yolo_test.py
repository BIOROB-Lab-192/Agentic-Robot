from ultralytics import YOLO

# Initialize model
model = YOLO("./models/yoloe-26x-seg.pt")

model.set_classes(["pepper", "bell pepper", "trench coat", "coat"])

# Run prediction. No prompts required.
results = model.predict("./test_images/coat.JPG")

# Show results
results[0].show()
