import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers
from datetime import datetime, timedelta
import os
import shutil
import matplotlib.pyplot as plt


# check if all images in trains, validation, test have the same resolution
def check_images_resolution():
    resolutions = set()  # we add resolutions to this set and at the end it needs to have size 1
    stages = ['train', 'validation', 'test']

    for stage in stages:
        csv = pd.read_csv(f'{stage}.csv')
        for i, row in csv.iterrows():
            image_id = row['image_id']
            image_path = f'{stage}/{image_id}.png'
            image = Image.open(image_path)
            resolutions.add(image.size)

    if len(resolutions) != 1:
        raise Exception('check_images_resolution(): there are some images in the dataset '
                        'that don\'t have the standard resolution (100x100)')

    return True


# returns numpy arrays of images and labels from a given stage (train, validation, test)
def load_images_and_labels(stage):
    if stage not in ['train', 'validation', 'test']:
        raise Exception('load_images(): invalid stage name')

    images = []
    labels = []
    csv = pd.read_csv(f'{stage}.csv')

    for i, row in csv.iterrows():  # iterrows() lets us iterate over the rows of the csv
        # extract the image from csv
        image_id = row['image_id']  # extract the id
        image_path = f'{stage}/{image_id}.png'  # extract the image path
        image = Image.open(image_path)  # open the image
        image_array = np.array(image)  # convert to array
        images.append(image_array)  # add to the list

        # extract the label from csv only for training and validation
        if stage != 'test':
            label = row['label']  # extract label
            labels.append(label)  # add to the list

    # if it's the test stage, return none for the labels
    if stage == 'test':
        return np.array(images), None

    return np.array(images), np.array(labels)


def preprocess_images_and_labels(images, labels=None):
    # convert labels to categorical, if given
    categorized_labels = None
    if labels is not None:
        categorized_labels = to_categorical(labels, 5)  # convert to categorical with 5 classes

    # normalize images to [0, 1]
    normalized_images = images / 255.0

    return normalized_images, categorized_labels


def deepfake_classification_cnn_model():
    cnn_model = Sequential()

    # define shape layer
    cnn_model.add(Input(shape=(100, 100, 3)))

    # image augumentation
    # i currently have overfitting so i try image augumentation
    cnn_model.add(layers.RandomFlip('horizontal'))
    cnn_model.add(layers.RandomRotation(0.05))

    # CONVOLUTIUONAL BLOCK 1
    # layer - relu removes negative values; input_shape 100x100 w/ RGB
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # input_shape = (100, 100, 3)
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # input_shape = (100, 100, 3)
    # pooling layer reduces dimensions - downsampling
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # add dropout to reduce overfitting by making the model not rely on a specific feature
    cnn_model.add(Dropout(0.25))

    # CONVOLUTIONAL BLOCK 2
    cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # input_shape = (50, 50, 64)
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # input_shape = (50, 50, 64)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))

    # CONVOLUTIONAL BLOCK 3
    cnn_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))  # input_shape = (25, 25, 128)
    cnn_model.add(BatchNormalization())
    cnn_model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.5))

    # convert results from array to vector (of size 12 x 12 x 256 = 36864)
    cnn_model.add(Flatten())
    # take all the features from the convolutional blocks and turn them into final predicitions - classifier
    # l2 regularization to keep weights small and prevent overfitting
    cnn_model.add(Dense(units=256, activation='relu', kernel_regularizer=l2(0.001)))
    cnn_model.add(Dropout(0.5))  # too many weights (36864 x 256) that can cause overfitting
    # another layer for more refining
    cnn_model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))
    cnn_model.add(Dropout(0.3))  # too many weights again (256 x 128)
    # the 5 classes of the deepfake classification, softmax converts scores to probs
    cnn_model.add(Dense(5, activation='softmax'))

    print(f'\n{40 * '='} CNN MODEL SHAPE {40 * '='}\n')
    print(cnn_model.summary())
    print(f'\n{40 * '='} CNN MODEL SHAPE {40 * '='}\n')

    return cnn_model


def prepare_model_for_training(cnn_model):
    """
    1. optimizer=i read that adam is common for cnn
    2. metrics=accuracy percentage is intuituive and works best because the classes in the training,
        images are distributed evenly
    3. loss=categorical_crossentropy suitable for 5 classes that are I converted to categorical,
        and it is good with softmax activation

    """
    cnn_model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
    return cnn_model


def train_cnn_model(cnn_model, train_images, train_labels, validation_images, validation_labels, save_directory):
    """
    batch size TODO
    epochs set to 50 but the earlystopping makes it stop at about 25
    save the best model with modelcheckpoint callback
    earlystopping checks the validation loss and if it does not improve for 5 epochs it ends the training,
        keeping the best weights
    i use tensorboard for quick in browser performance checking
    """
    cnn_model.fit(
        x=train_images, y=train_labels, batch_size=256, epochs=120,
        validation_data=(validation_images, validation_labels), # class_weight={0:0.95, 1:1.4, 2:1.1, 3:0.7, 4:1.8},
        callbacks=[ModelCheckpoint(filepath=f'{save_directory}/best_deepfake_classification_cnn_model.keras'),
                   EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                   TensorBoard(log_dir=f'{save_directory}/TensorBoard_logs')
                   ]
    )


def get_and_create_save_directory(running_on_vps=False):
    if running_on_vps:
        current_time_vps = datetime.now()
        current_time_romania = current_time_vps + timedelta(hours=3)  # vps is 3h behind
        run_time = current_time_romania.strftime('%d_%m_%Y-%H:%M')
    else:
        run_time = datetime.now().strftime('%d_%m_%Y-%H:%M')  # get the run time of training to organize for submission
    run_directory_path = f'runs/{run_time}'  # the file path in which this model will be saved completed with the timestamp
    os.makedirs(run_directory_path, exist_ok=True)  # create the actual directory of saving
    shutil.copy('main.py', f'{run_directory_path}/main.py')  # copy script so I can go back and look at the params
    return run_directory_path, run_time  # returns run_time as well so i can use it in the confusion matrix plot title


def main():
    try:
        # before starting, check that images are brought to a given standard
        print(f'\n{15 * '>'} CHECKING IMAGES RESOLUTION {15 * '<'}\n')
        check_images_resolution()

        # load training, validation, test images and labels
        print(f'\n{15 * '>'} LOADING IMAGES {15 * '<'}\n')
        train_images, train_labels = load_images_and_labels('train')
        validation_images, validation_labels = load_images_and_labels('validation')
        test_images, test_labels = load_images_and_labels('test')

        # preprocess images and labels
        print(f'\n{15 * '>'} PREPROCESS IMAGES {15 * '<'}\n')
        train_images, train_labels = preprocess_images_and_labels(train_images, train_labels)
        validation_images, validation_labels = preprocess_images_and_labels(validation_images, validation_labels)
        test_images, test_labels = preprocess_images_and_labels(test_images)

        print(f'\n{40 * '='} DATASET SHAPE {40 * '='}\n')
        print('Train images shape ', train_images.shape)
        print('Train labels shape ', train_labels.shape)
        print('Validation images shape ', validation_images.shape)
        print('Validation labels shape ', validation_labels.shape)
        print('Test images shape ', test_images.shape)
        print(f'\n{40 * '='} DATASET SHAPE {40 * '='}\n')

        # create and get the file path of the save directory
        save_directory, run_time = get_and_create_save_directory(True)
        evaluation_file = open(f'{save_directory}/training_evaluation.txt', 'a')

        # get the model
        print(f'\n{15 * '>'} BUILDING MODEL {15 * '<'}\n')
        cnn_model = deepfake_classification_cnn_model()
        # prepare the model
        print(f'\n{15 * '>'} PREPARING MODEL {15 * '<'}\n')
        cnn_model = prepare_model_for_training(cnn_model)
        # train the model
        print(f'\n{15 * '>'} TRAINING MODEL {15 * '<'}\n')
        train_cnn_model(cnn_model, train_images, train_labels, validation_images, validation_labels, save_directory)

        # extract the training and validation accuracy and loss
        # I am not using the value returned by the cnn_model.train(), because i have early stopping callback
        training_loss, training_accuracy = cnn_model.evaluate(train_images, train_labels, verbose=0)
        validation_loss, validation_accuracy = cnn_model.evaluate(validation_images, validation_labels, verbose=0)
        print(f'\n{40 * '='} CNN MODEL TRAINING EVALUATION {40 * '='}\n')
        print('Training accuracy: ', training_accuracy)
        print('Training loss: ', training_loss)
        print('Validation accuracy: ', validation_accuracy)
        print('Validation loss: ', validation_loss)
        print(f'Accuracy gap (< 0.05 - 0.1): {training_accuracy - validation_accuracy}')
        print(f'\n{40 * '='} CNN MODEL TRAINING EVALUATION {40 * '='}\n')
        evaluation_file.write(f'\n{40 * "="} CNN MODEL TRAINING EVALUATION {40 * "="}\n')
        evaluation_file.write(f'\nTraining accuracy: {training_accuracy}\n')
        evaluation_file.write(f'Training loss: {training_loss}\n')
        evaluation_file.write(f'Validation accuracy: {validation_accuracy}\n')
        evaluation_file.write(f'Validation loss: {validation_loss}\n')
        evaluation_file.write(f'Accuracy gap (< 0.05 - 0.1): {training_accuracy - validation_accuracy}\n')
        evaluation_file.write(f'\n{40 * "="} CNN MODEL TRAINING EVALUATION {40 * "="}\n')
        
        # test the model
        print(f'\n{15 * '>'} TESTING MODEL {15 * '<'}\n')
        test = cnn_model.predict(test_images)
        test_labels = np.argmax(test, axis=1)

        # save results
        print(f'\n{15 * '>'} SAVING RESULTS {15 * '<'}\n')
        submission_csv = pd.read_csv(f'test.csv')
        submission_csv['label'] = test_labels
        submission_csv[['image_id', 'label']].to_csv(f'{save_directory}/submission.csv', index=False)

        # get confusion matrix
        print(f'\n{15 * '>'} SAVING CONFUSION MATRIX AT {save_directory}/confusion_matrix.png {15 * '<'}\n')
        validation = cnn_model.predict(validation_images)
        validation_predicted_labels = np.argmax(validation, axis=1)
        validation_real_labels = np.argmax(validation_labels, axis=1)
        validation_confusion_matrix = confusion_matrix(validation_real_labels, validation_predicted_labels)
        # make a plot with heatmap for better represenation
        plt.figure(figsize=(10, 10))
        classes = [0, 1, 2, 3, 4]
        confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=validation_confusion_matrix, display_labels=classes)
        confusion_matrix_display.plot(cmap='Greens', values_format='d')  # 'd' for integers
        plt.savefig(f'{save_directory}/confusion_matrix.png')

    except Exception as e:
        print('Exception in main(): ', e)
        return


if __name__ == "__main__":
    main()

"""
TODO:
    change cnn layers structure 
"""
