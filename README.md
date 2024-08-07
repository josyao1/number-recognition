# Handwritten Digit Recognition

This project aims to recognize handwritten digits using a Convolutional Neural Network (CNN) and a web interface built with Flask. The project includes a drawing interface where users can draw digits, and the system will predict the digit using a trained neural network model.

## Project Structure

```
Number Recognition/
├── app.py
├── templates/
│   └── index.html
├── train_and_save_model.py
├── recognizer.py
├── pygame_frontend.py
├── digit_recognizer_model.h5
├── requirements.txt
└── README.md
```

## Neural Network Overview

The neural network used in this project is a Convolutional Neural Network (CNN) designed to handle image data effectively. The CNN architecture includes multiple convolutional layers followed by pooling layers, and finally fully connected layers to output the predicted digit.

### Model Architecture

- **Input Layer**: 28x28 grayscale images with a single channel.
- **Conv2D Layer 1**: 64 filters, kernel size 3x3, ReLU activation.
- **Conv2D Layer 2**: 32 filters, kernel size 3x3, same padding, ReLU activation.
- **MaxPooling2D Layer 1**: Pool size 2x2.
- **Conv2D Layer 3**: 16 filters, kernel size 3x3, same padding, ReLU activation.
- **MaxPooling2D Layer 2**: Pool size 2x2.
- **Conv2D Layer 4**: 64 filters, kernel size 3x3, same padding, ReLU activation.
- **MaxPooling2D Layer 3**: Pool size 2x2.
- **Flatten Layer**: Flatten the output from the convolutional layers.
- **Dense Layer 1**: 128 units, ReLU activation.
- **Dense Layer 2**: 10 units, sigmoid activation.

The model is compiled with the Adam optimizer and Sparse Categorical Crossentropy loss function. The accuracy metric is used to evaluate the model performance.

## Setup and Usage

### Prerequisites

- Python 3.6 or higher
- Conda (recommended) or virtualenv

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/number-recognition.git
   cd number-recognition
   ```

2. **Create and activate a virtual environment**:

   Using Conda:
   ```bash
   conda create --name number-recognition python=3.8
   conda activate number-recognition
   ```

   Using virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**:

   Run the training script to train the neural network and save the model:

   ```bash
   python train_and_save_model.py
   ```

### Running the Application

1. **Start the Flask server**:

   ```bash
   python app.py
   ```

   The Flask server will start and listen on `http://127.0.0.1:5000`.

2. **Access the web interface**:

   Open your browser and navigate to `http://127.0.0.1:5000`. You will see a drawing interface where you can draw digits.

3. **Predict digits**:

   - Draw a digit in the canvas.
   - Click the "Predict" button to send the drawing to the server for prediction.
   - The predicted digit will be displayed on the page.

### Using Pygame Front End

If you prefer using a local Pygame front end for drawing digits:

1. **Run the Pygame front end**:

   ```bash
   python pygame_frontend.py
   ```

   A Pygame window will open where you can draw digits and get predictions.

## Reasoning Behind the Neural Network

The chosen Convolutional Neural Network (CNN) architecture is well-suited for image recognition tasks due to its ability to capture spatial hierarchies in images. Convolutional layers help in detecting features like edges, corners, and textures, while pooling layers reduce dimensionality and computation. Fully connected layers at the end combine these features to predict the digit.

The architecture is simple yet effective for the MNIST dataset, providing a good balance between complexity and performance. The model can be further tuned and improved with additional layers or hyperparameter adjustments.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

This `README.md` file provides a comprehensive overview of the project, including the neural network architecture, setup instructions, and usage guidelines. Let me know if you need any further adjustments or additional information!