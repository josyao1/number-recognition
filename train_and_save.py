import tensorflow as tf
from tensorflow.keras import layers, models

def train_and_save_model():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape the data to include the channel dimension
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Create the model
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Loss: {loss}")

    # Save the model
    model.save('digit_recognizer_model.h5')

if __name__ == "__main__":
    train_and_save_model()
