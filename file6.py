import cv2
from pathlib import Path
import numpy as np


def extract_graphs(image_path, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    cv2.imshow("dark grey", gray)
    cv2.waitKey(0)

    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(gray, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(gray, bg, scale=255)
    #out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 

    cv2.imshow('binary', out_gray)
    cv2.waitKey(0)

    # Apply threshold to get binary image
    _, binary = cv2.threshold(out_gray, 230, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    img_countours = image.copy()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(img_countours, contours, -1, (0, 255, 0), 1)
    cv2.imshow("image with contours", image_with_contours)
    cv2.waitKey(0)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image_with_contours, kernel, iterations=1)

    cv2.imshow("image with contours eroted", erosion)
    cv2.waitKey(0)
    # Apply dilation and display
    gray = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image with contours grayed", gray)
    cv2.waitKey(0)

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    #_, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("image with contours grayed bin", binary)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(img_countours, contours, -1, (0, 255, 0), 1)
    cv2.imshow("image with contours after", image_with_contours)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Minimum area threshold to filter out noise
    min_area = 1000
    
    # Aspect ratio thresholds (you can adjust these based on your needs)
    min_width = 30  # Minimum acceptable width
    min_height = 30  # Minimum acceptable height
    max_aspect_ratio = 3  # Maximum width-to-height ratio for a proper rectangle/square
    min_aspect_ratio = 0.4  # Minimum acceptable width-to-height ratio (to prevent too narrow graphs)

    # Process each contour
    for idx, contour in enumerate(contours):
        # Calculate contour area
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Reject graphs that are too narrow or too small
            if w < min_width or h < min_height:
                continue

            # Check the aspect ratio (width / height) to ensure it's a proper rectangle
            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # Add padding around the graph (optional)
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Extract the graph region
            graph = image[y:y+h, x:x+w]
            
            # Save the extracted graph
            output_path = f"{output_dir}/graph_{idx}.png"
            cv2.imwrite(output_path, graph)
            print(f"Saved graph {idx} to {output_path}")

image_path = "img/Retail_Sales_Distribution_Network_Dashboard_With_Forecasting_3 (1).jpg"
image_path = "img/Predictive_Modeling_Model_Performance_1.jpg"
#image_path = "img/Revenue_Dashboard_2.jpg"
output_dir = "out"

extract_graphs(image_path, output_dir)
