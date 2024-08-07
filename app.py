from flask import Flask, render_template, request, jsonify
from recognizer import Recognizer
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Initialize the recognizer
recognizer = Recognizer('updated_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image']
    img_data = base64.b64decode(img_data.split(',')[1])
    
    img = Image.open(io.BytesIO(img_data)).convert('L')
    img = img.resize((28, 28))
    img = np.array(img).reshape(1, 28, 28)
    img = np.invert(img) / 255.0
    
    print(f"Processed image shape: {img.shape}")
    print(f"Processed image data: {img}")
    
    prediction = recognizer.model.predict(img)
    predicted_digit = int(np.argmax(prediction))
    
    print(f"Model prediction: {prediction}")
    print(f"Predicted digit: {predicted_digit}")

    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
