import cv2
import numpy as np
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt


class ImageEnhancer:
    """Class for various image enhancement operations."""
    
    def __init__(self, image_path):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        self.image = self.original.copy()
    
    def brightness(self, factor):
        """Adjust brightness (0.5-2.0 recommended)."""
        pil_img = Image.fromarray(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        return cv2.cvtColor(np.array(enhancer.enhance(factor)), cv2.COLOR_RGB2BGR)
    
    def contrast(self, factor):
        """Adjust contrast (0.5-2.0 recommended)."""
        pil_img = Image.fromarray(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        return cv2.cvtColor(np.array(enhancer.enhance(factor)), cv2.COLOR_RGB2BGR)
    
    def saturation(self, factor):
        """Adjust color saturation (0.5-2.0 recommended)."""
        pil_img = Image.fromarray(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(pil_img)
        return cv2.cvtColor(np.array(enhancer.enhance(factor)), cv2.COLOR_RGB2BGR)
    
    def blur(self, kernel_size=5):
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(self.original, (kernel_size, kernel_size), 0)
    
    def sharpen(self):
        """Sharpen the image."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 1.0
        return cv2.filter2D(self.original, -1, kernel)
    
    def histogram_equalization(self):
        """Enhance contrast using histogram equalization."""
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    def edge_detection(self):
        """Detect edges using Canny."""
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    
    def denoise(self):
        """Remove noise using bilateral filter."""
        return cv2.bilateralFilter(self.original, 9, 75, 75)
    
    def rotate(self, angle):
        """Rotate image by angle (degrees)."""
        h, w = self.original.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(self.original, matrix, (w, h))
    
    def resize(self, scale):
        """Resize image by scale factor."""
        h, w = self.original.shape[:2]
        return cv2.resize(self.original, (int(w*scale), int(h*scale)))


def display_comparison(original, enhanced, title="Comparison"):
    """Display original and enhanced images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()