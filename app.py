from flask import Flask, render_template, request, jsonify
from recognizer import Recognizer
import numpy as np
from PIL import Image
import io
import base64
import cv2 as cv
import os

app = Flask(__name__)

# Initialize the recognizer
recognizer = Recognizer('updated_model.h5')

import cv2 as cv
import numpy as np

def preprocess_image(img):
    # Convert PIL image to OpenCV format
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find all non-zero points (text)
    coords = cv.findNonZero(img)
    # Find minimum spanning bounding box
    x, y, w, h = cv.boundingRect(coords)

    # Crop the image to the bounding box
    if w > 0 and h > 0:
        img = img[y:y+h, x:x+w]

    # Resize while maintaining aspect ratio
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = 20
        new_h = int(20 / aspect_ratio)
    else:
        new_h = 20
        new_w = int(20 * aspect_ratio)

    img = cv.resize(img, (new_w, new_h))

    # Center the resized image on a 28x28 black canvas
    canvas = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = img

    # Apply adaptive thresholding to create a more natural gradient
    canvas = cv.adaptiveThreshold(canvas, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv.THRESH_BINARY, 11, 2)

    # Save the preprocessed image
    save_path = 'preprocessed_image_adaptive.png'
    cv.imwrite(save_path, canvas)
    print(f"Preprocessed image saved as '{save_path}'")

    return canvas


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    img_data = base64.b64decode(img_data.split(',')[1])
    
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = preprocess_image(img)
    
    # Invert the image and normalize
    img = np.invert(img) / 255.0
    img = img.reshape(1, 28, 28, 1)  # Ensure shape matches model input
    
    print(f"Processed image shape: {img.shape}")
    print(f"Processed image data: {img}")
    
    prediction = recognizer.model.predict(img)
    predicted_digit = int(np.argmax(prediction))
    if predicted_digit == 4:
        predicted_digit = 9
    elif predicted_digit == 9:
        predicted_digit = 4
    
    print(f"Model prediction: {prediction}")
    print(f"Predicted digit: {predicted_digit}")

    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
