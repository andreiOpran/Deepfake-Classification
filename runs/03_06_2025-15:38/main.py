import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
import os
import shutil


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

    # CONVOLUTIUONAL BLOCK 1
    # layer - relu removes negative values; input_shape 100x100 w/ RGB
    cnn_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # input_shape = (100, 100, 3)
    # pooling layer reduces dimensions - downsampling
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    # add dropout to reduce overfitting by making the model not rely on a specific feature
    cnn_model.add(Dropout(0.25))

    # CONVOLUTIONAL BLOCK 2
    cnn_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # input_shape = (50, 50, 32)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))

    # CONVOLUTIONAL BLOCK 3
    cnn_model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # input_shape = (25, 25, 64)
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.5))

    # convert results from array to vector (of size 12 x 12 x 128 = 18432)
    cnn_model.add(Flatten())
    # take all the features from the convolutional blocks and turn them into final predicitions - classifier
    cnn_model.add(Dense(units=128, activation='relu'))
    cnn_model.add(Dropout(0.5))  # too many weights (18432 x 128) that can cause overfitting
    # the 5 classes of the deepfake classification, softmax converts scores to probs
    cnn_model.add(Dense(5, activation='softmax'))

    print(f'\n{40 * '='} CNN MODEL SHAPE {40 * '='}\n')
    print(cnn_model.summary())
    print(f'\n{40 * '='} CNN MODEL SHAPE {40 * '='}\n')

    return cnn_model


def prepare_model_for_training(cnn_model):
    """
    1. optimizer=sgd is slow but provides good results
    2. metrics=accuracy percentage is intuituive and works best because the classes in the training,
        images are distributed evenly
    3. loss=categorical_crossentropy suitable for 5 classes that are I converted to categorical,
        and it is good with softmax activation

    """
    cnn_model.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')
    return cnn_model


def train_cnn_model(cnn_model, train_images, train_labels, validation_images, validation_labels, save_directory):
    cnn_model.fit(
        x=train_images, y=train_labels, batch_size=16, epochs=3,
        validation_data=(validation_images, validation_labels),
        callbacks=[ModelCheckpoint(filepath=f'{save_directory}/best_deepfake_classification_cnn_model.h5'),
                   EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                   TensorBoard(log_dir=f'{save_directory}/TensorBoard_logs')
        ]
    )


def get_and_create_save_directory():
    run_time = datetime.now().strftime('%d_%m_%Y-%H:%M')  # get the run time of training to organize for submission
    run_directory_path = f'runs/{run_time}'  # the file path in which this model will be saved completed with the timestamp
    os.makedirs(run_directory_path, exist_ok=True)  # create the actual directory of saving
    shutil.copy('main.py', f'{run_directory_path}/main.py')  # copy script so I can go back and look at the params
    return run_directory_path


def main():
    try:
        # before starting, check that images are brought to a given standard
        check_images_resolution()

        # load training, validation, test images and labels
        train_images, train_labels = load_images_and_labels('train')
        validation_images, validation_labels = load_images_and_labels('validation')
        test_images, test_labels = load_images_and_labels('test')

        # preprocess images and labels
        train_images, train_labels = preprocess_images_and_labels(train_images, train_labels)
        validation_images, validation_labels = preprocess_images_and_labels(validation_images, validation_labels)
        test_images, test_labels = preprocess_images_and_labels(test_images)

        print(f'\n{'=' * 50}')
        print('Train images shape ', train_images.shape)
        print('Train labels shape ', train_labels.shape)
        print('Validation images shape ', validation_images.shape)
        print('Validation labels shape ', validation_labels.shape)
        print('Test images shape ', test_images.shape)
        print(f'{'=' * 50}\n')

        # create and get the file path of the save directory
        save_directory = get_and_create_save_directory()

        # get the model
        cnn_model = deepfake_classification_cnn_model()
        # prepare the model
        cnn_model = prepare_model_for_training(cnn_model)
        # train the model
        train_cnn_model(cnn_model, train_images, train_labels, validation_images, validation_labels, save_directory)
        # get the loss and accuracy of the validation set
        validation_loss, validation_accuracy = cnn_model.evaluate(validation_images, validation_labels, verbose=0)
        print('Validation accuracy: ', validation_accuracy)

        # test the model
        test = cnn_model.predict(test_images)
        test_labels = np.argmax(test, axis=1)

        # save results
        submission_csv = pd.read_csv(f'test.csv')
        submission_csv['label'] = test_labels
        submission_csv[['image_id', 'label']].to_csv(f'{save_directory}/submission.csv', index=False)

    except Exception as e:
        print('Exception in main(): ', e)
        return


if __name__ == "__main__":
    main()
