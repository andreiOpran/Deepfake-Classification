import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


def check_images_resolution():
    pass


def load_images_and_labels(stage):
    pass


def preprocess_data(images, labels=None):
    """Normalize images and convert labels to categorical"""
    # TODO: Normalize images (divide by 255.0)
    # TODO: If labels exist, convert to categorical (one-hot encoding)
    # TODO: Ensure proper shape for CNN input
    pass


def build_cnn_model(input_shape, num_classes=5):
    """Build CNN architecture from scratch"""
    model = Sequential()

    # TODO: Add Conv2D layers with appropriate filters, kernel size, activation
    # TODO: Add MaxPooling2D layers for downsampling
    # TODO: Add Dropout layers for regularization
    # TODO: Add Flatten layer
    # TODO: Add Dense layers
    # TODO: Add final output layer with softmax activation

    return model


def compile_model(model):
    """Compile the model with optimizer, loss, and metrics"""
    # TODO: Use Adam optimizer
    # TODO: Use categorical_crossentropy loss
    # TODO: Include accuracy metric
    pass


def train_model(model, train_data, train_labels, val_data, val_labels, epochs=50, batch_size=32):
    """Train the CNN model"""
    # TODO: Set up callbacks (EarlyStopping, ModelCheckpoint)
    # TODO: Fit the model
    # TODO: Return training history
    pass


def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance"""
    # TODO: Make predictions
    # TODO: Calculate accuracy, confusion matrix, classification report
    pass


def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    # TODO: Create plots for accuracy and loss over epochs
    pass


def main():
    try:
        check_images_resolution()
    except Exception as e:
        print(e)
        return

    # Load data
    try:
        train_images, train_labels = load_images_and_labels('train')
        validation_images, validation_labels = load_images_and_labels('validation')
        test_images, test_labels = load_images_and_labels('test')
    except Exception as e:
        print(e)
        return

    print(f"Train images shape: {train_images.shape}")
    print(f"Validation images shape: {validation_images.shape}")
    print(f"Test images shape: {test_images.shape}")

    # TODO: Preprocess data
    # train_images_processed, train_labels_processed = preprocess_data(train_images, train_labels)
    # val_images_processed, val_labels_processed = preprocess_data(validation_images, validation_labels)
    # test_images_processed, _ = preprocess_data(test_images)

    # TODO: Build and compile model
    # input_shape = (100, 100, 3)  # or (100, 100, 1) for grayscale
    # model = build_cnn_model(input_shape)
    # compile_model(model)

    # TODO: Train model
    # history = train_model(model, train_images_processed, train_labels_processed,
    #                      val_images_processed, val_labels_processed)

    # TODO: Plot training history
    # plot_training_history(history)

    # TODO: Evaluate on test set
    # evaluate_model(model, test_images_processed, test_labels)

    # TODO: Save final model
    # model.save('deepfake_cnn_model.h5')


if __name__ == "__main__":
    main()