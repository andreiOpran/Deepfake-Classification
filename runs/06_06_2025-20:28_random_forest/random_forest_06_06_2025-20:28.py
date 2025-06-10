import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
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
    # normalize images to [0, 1]
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(normalized_images.shape[0], -1)
    return flattened_images, labels


def deepfake_classification_model():
    model = RandomForestClassifier(n_estimators=500, max_depth=20, n_jobs=-1, verbose=2)
    return model


def train_model(model, train_images, train_labels):
    model.fit(train_images, train_labels)


def get_and_create_save_directory(running_on_vps=False):
    if running_on_vps:
        current_time_vps = datetime.now()
        current_time_romania = current_time_vps + timedelta(hours=3)  # vps is 3h behind
        run_time = current_time_romania.strftime('%d_%m_%Y-%H:%M')
    else:
        run_time = datetime.now().strftime('%d_%m_%Y-%H:%M')  # get the run time of training to organize for submission
    run_directory_path = f'runs/{run_time}_random_forest'  # the file path in which this model will be saved completed with the timestamp
    os.makedirs(run_directory_path, exist_ok=True)  # create the actual directory of saving
    shutil.copy('random_forest.py', f'{run_directory_path}/random_forest.py')  # copy script so I can go back and look at the params
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
        save_directory, run_time = get_and_create_save_directory()
        evaluation_file = open(f'{save_directory}/training_evaluation.txt', 'a')

        # get the model
        print(f'\n{15 * '>'} BUILDING MODEL {15 * '<'}\n')
        model = deepfake_classification_model()

        # train the model
        print(f'\n{15 * '>'} TRAINING MODEL {15 * '<'}\n')
        train_model(model, train_images, train_labels)

        train_predictions = model.predict(train_images)
        validation_predictions = model.predict(validation_images)

        train_accuracy = accuracy_score(train_labels, train_predictions)
        validation_accuracy = accuracy_score(validation_labels, validation_predictions)

        print(f'\n{40 * "="} RANDOM FOREST TRAINING EVALUATION {40 * "="}\n')
        print('Training accuracy: ', train_accuracy)
        print('Validation accuracy: ', validation_accuracy)
        print(f'Accuracy gap: {train_accuracy - validation_accuracy}')
        print(f'\n{40 * "="} RANDOM FOREST TRAINING EVALUATION {40 * "="}\n')
        evaluation_file.write(f'\n{40 * "="} RANDOM FOREST TRAINING EVALUATION {40 * "="}\n\n')
        evaluation_file.write(f'Training accuracy: {train_accuracy}\n')
        evaluation_file.write(f'Validation accuracy: {validation_accuracy}\n')
        evaluation_file.write(f'Accuracy gap: {train_accuracy - validation_accuracy}\n')
        evaluation_file.write(f'\n{40 * "="} RANDOM FOREST TRAINING EVALUATION {40 * "="}\n')

        # test the model
        print(f'\n{15 * '>'} TESTING MODEL {15 * '<'}\n')
        test = model.predict(test_images)

        # save results
        print(f'\n{15 * '>'} SAVING RESULTS {15 * '<'}\n')
        submission_csv = pd.read_csv(f'test.csv')
        submission_csv['label'] = test
        submission_csv[['image_id', 'label']].to_csv(f'{save_directory}/submission_{run_time}.csv', index=False)

        # get confusion matrix
        print(f'\n{15 * '>'} SAVING CONFUSION MATRIX AT {save_directory}/confusion_matrix.png {15 * '<'}\n')
        validation_confusion_matrix = confusion_matrix(validation_labels, validation_predictions)
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

