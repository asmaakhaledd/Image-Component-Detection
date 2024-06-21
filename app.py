import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

# Loading the YOLOv5 model
model = YOLO('yolov5su.pt')  

# Detecting objects in the image
def analyze_image(image):
    results = model(image)
    result = results[0]  # Access the first result (for a single image)
    
    # Extract bounding boxes, labels, and confidences
    bbox = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    labels = [model.names[label] for label in labels]
    
    # Render the image with bounding boxes
    output_image = np.squeeze(result.plot())
    return output_image, labels

# Streamlit app title
st.title("Image Component Detection")

# Image upload widget accept only jpg, jpeg, and png files
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Analyze image button
if st.button("Analyse Image"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Analyze the image
        output_image, labels = analyze_image(image_np)

        # Display the image with detections
        st.image(output_image, caption='Processed Image', use_column_width=True)

        # Display the results
        st.sidebar.write("Detected Components:")
        for label in labels:
            st.sidebar.write(f"- {label}")

    else:
        st.write("Please upload an image first.")
