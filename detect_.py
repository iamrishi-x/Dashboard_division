import cv2
from pathlib import Path

def extract_graphs(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    img_countours = image.copy()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(img_countours, contours, -1, (0, 255, 0), 1)

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

# image_path = "img/Retail_Sales_Distribution_Network_Dashboard_With_Forecasting_3 (1).jpg"
image_path = "img/Predictive_Modeling_Model_Performance_1.jpg"
#image_path = "img/Revenue_Dashboard_2.jpg"
output_dir = "out"

extract_graphs(image_path, output_dir)
