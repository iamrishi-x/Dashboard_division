import numpy as np
import cv2
from keras.models import load_model

# Load the trained model
model_path = '/Users/user/Documents/Project/___e-zest___/Dashboard_division/model/graph_recognisor2.h5'
model = load_model(model_path)

# Preprocessing function (same as in training)
def preprocessing(img):
    # Check if image is already grayscale (i.e., single channel)
    if len(img.shape) == 3:  # 3 channels (BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.equalizeHist(img)  # Enhance contrast
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to predict the class for an ROI
def predict_roi(roi, model):
    # Resize ROI to model's expected input size
    roi = cv2.resize(roi, (32, 32))
    
    # Preprocess the ROI
    roi = preprocessing(roi)
    
    # Reshape for model input (add batch and channel dimensions)
    roi = roi.reshape(1, 32, 32, 1)
    
    # Predict class
    predictions = model.predict(roi)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_index, confidence

# Detect objects (ROIs) in the image
def detect_and_draw_boxes(image_path, model):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image not found at {image_path}")
        return
    
    # Convert to grayscale for ROI detection (assume digits are prominent)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # Binary threshold
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and predict each ROI
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip small regions (likely noise)
        if w < 10 or h < 10:
            continue

        # Extract the ROI
        roi = gray[y:y+h, x:x+w]

        # Predict the class of the ROI
        class_index, confidence = predict_roi(roi, model)
        
        # Draw bounding box and label on the original image
        label = f"{class_index} ({confidence:.2f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue label
    
    # Display the result
    cv2.imshow("Detected Objects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image you want to test
test_image_path = '/Users/user/Documents/Project/___e-zest___/Dashboard_division/img/Retail_Sales_Distribution_Network_Dashboard_With_Forecasting_3 (1).jpg'

# Call the function
detect_and_draw_boxes(test_image_path, model)
