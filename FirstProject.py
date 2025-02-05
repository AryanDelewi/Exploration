# %% Import required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# %% Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images to scale pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# %% Define the neural network model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a 1D vector
    layers.Dense(128, activation='relu'),  # First hidden layer (128 neurons)
    layers.Dense(64, activation='relu'),   # Second hidden layer (64 neurons)
    layers.Dense(10, activation='softmax') # Output layer (10 neurons for classification)
])

# %% Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %% Train the model
model.fit(x_train, y_train, epochs=3)

# %% Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# %% Make a prediction on a sample image
sample_image = np.expand_dims(x_test[0], axis=0)  # Reshape for prediction
prediction = model.predict(sample_image)
predicted_label = np.argmax(prediction)

# Show the image with the predicted label
plt.imshow(x_test[0], cmap="gray")
plt.title(f"Predicted: {predicted_label}, True: {y_test[0]}")
plt.axis("off")
plt.show()