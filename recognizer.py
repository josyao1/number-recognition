import cv2 as cv
import numpy as np
import tensorflow as tf

class Recognizer():
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, img):
        coords = cv.findNonZero(img)  # Find all non-zero points (text)
        x, y, w, h = cv.boundingRect(coords)  # Find minimum spanning bounding box

        if w > 0 and h > 0:
            img = img[y:y+h, x:x+w]

        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = 20
            new_h = int(20 / aspect_ratio)
        else:
            new_h = 20
            new_w = int(20 * aspect_ratio)

        img = cv.resize(img, (new_w, new_h))

        canvas = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = img

        return canvas

    def guess(self, img_path):
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = self.preprocess_image(img)
        img = np.invert(np.array([img])) / 255.0

        print(f"Image shape after inversion: {img.shape}")

        if img.shape != (1, 28, 28):
            print("Error: Image shape is incorrect. Expected (1, 28, 28)")
            return None

        prediction = self.model.predict(img)

        if prediction is None:
            print("Error: Model prediction returned None")
            return None

        return np.argmax(prediction)

if __name__ == "__main__":
    recognizer = Recognizer('updated_model.h5')
    result = recognizer.guess('images/1a.jpg')
    if result is not None:
        print(f"Predicted digit: {result}")
    else:
        print("Prediction failed")
