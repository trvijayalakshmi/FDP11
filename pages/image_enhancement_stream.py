import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Medical Image Enhancement",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Image Enhancement using Streamlit")
st.write("Upload a medical image (X-ray, MRI, CT, Ultrasound) and apply enhancement techniques.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a medical image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, caption="Original Image", use_container_width=True)

    # Sidebar controls
    st.sidebar.header("Enhancement Options")

    operation = st.sidebar.selectbox(
        "Choose Enhancement Technique",
        (
            "Histogram Equalization",
            "CLAHE (Adaptive Histogram)",
            "Gaussian Smoothing",
            "Median Filtering",
            "Sharpening",
            "Edge Enhancement",
            "Brightness & Contrast"
        )
    )

    enhanced_img = img_np.copy()

    # Histogram Equalization
    if operation == "Histogram Equalization":
        enhanced_img = cv2.equalizeHist(img_np)

    # CLAHE
    elif operation == "CLAHE (Adaptive Histogram)":
        clip_limit = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0)
        tile_grid = st.sidebar.slider("Tile Grid Size", 4, 16, 8)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_grid, tile_grid)
        )
        enhanced_img = clahe.apply(img_np)

    # Gaussian Smoothing
    elif operation == "Gaussian Smoothing":
        kernel_size = st.sidebar.slider("Kernel Size (odd)", 3, 15, 5, step=2)
        enhanced_img = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)

    # Median Filtering
    elif operation == "Median Filtering":
        kernel_size = st.sidebar.slider("Kernel Size (odd)", 3, 15, 5, step=2)
        enhanced_img = cv2.medianBlur(img_np, kernel_size)

    # Sharpening
    elif operation == "Sharpening":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced_img = cv2.filter2D(img_np, -1, kernel)

    # Edge Enhancement
    elif operation == "Edge Enhancement":
        edges = cv2.Canny(img_np, 50, 150)
        enhanced_img = cv2.addWeighted(img_np, 0.8, edges, 0.2, 0)

    # Brightness & Contrast
    elif operation == "Brightness & Contrast":
        brightness = st.sidebar.slider("Brightness", -100, 100, 0)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
        enhanced_img = cv2.convertScaleAbs(
            img_np,
            alpha=contrast,
            beta=brightness
        )

    # Display enhanced image
    st.subheader("Enhanced Image")
    st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

    # Histogram comparison
    st.subheader("Histogram Comparison")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].hist(img_np.ravel(), bins=256, color="gray")
    ax[0].set_title("Original Histogram")

    ax[1].hist(enhanced_img.ravel(), bins=256, color="gray")
    ax[1].set_title("Enhanced Histogram")

    st.pyplot(fig)

else:
    st.info("Please upload a medical image to begin.")