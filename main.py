import os
import pathlib

import numpy as np
import pandas as pd
import psutil
from PIL import Image

# environment variable hacking to force Keras to use the Tensorflow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout

train_directory = pathlib.Path('train')
image_extension = 'tif'


def get_train_image(name, directory=train_directory):
    """
    Args:
        directory (pathlib.Path):
        name (str): The filename of the image to load, without the file extension or folder location

    Returns:
        4D numpy array (examples, height, width, channels)
    """
    img = Image.open(directory / (name + '.' + image_extension))
    img.load()
    return np.asarray(img, dtype="int32")


def load_data(file_names, max_files=10000):
    """
    Load in image files to a list of numpy arrays. Sequentially stack arrays
    into a single array.

    Args:
        file_names (list): A sequence of file names to load data from
        max_files (int): The maximum number of files to load, added due to performance issues

    Returns:

    """

    counter = 0

    # how often to print out memory usage and and file counter
    print_every = 100

    # how often to stack list of arrays into a single array
    stack_every = 10000

    arrays = []
    data = None

    for file_name in file_names:

        if counter % print_every == 0:
            print(f'loaded {counter} files')
            print("memory usage:", psutil.virtual_memory())
            print("swap memory usage:", psutil.swap_memory())

        # load image array and append to arrays
        arrays.append(get_train_image(name=file_name))

        if counter % stack_every == 0:
            if data is None:
                data = np.stack(arrays)
            else:
                data = np.concatenate([data] + [np.stack(arrays)], axis=0)
            arrays = []

        if counter >= max_files:
            if data is None:
                return np.stack(arrays)
            else:
                if len(arrays) > 0:
                    return np.concatenate([data] + [np.stack(arrays)], axis=0)
                return data

        counter += 1

    return np.concatenate([data] + [np.stack(arrays)], axis=0)


def normalize_image_data(data):
    # naive mean normalisation
    # TODO normalise on a per color channel basis

    return (data / 255.0) - 0.5


def unison_shuffled_copies(a, b):
    """
    Shuffles both arrays in sync along their first axis

    Args:
        a (Array): First array
        b (Array): Second array

    Returns:
        (a, b)

    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main():
    # loading labels in
    labels = pd.read_csv('train_labels.csv')

    # extracting pandas series as numpy arrays
    y = labels['label'].values
    file_names = labels['id'].values

    # shuffling labels and file names in sync
    y, file_names = unison_shuffled_copies(y, file_names)

    x = load_data(file_names)
    x = normalize_image_data(x)

    # trim labels to the same size as the number of loaded in files
    y = y[0:x.shape[0]:]

    # shuffling before splitting test and train sets
    y, x = unison_shuffled_copies(y, x)

    # set model training hyper parameters
    test_fraction = 0.1  # what fraction of examples to use in test set
    batch_size = 128  # mini batch size
    num_classes = 1
    epochs = 200
    dropout = 0.5  # fraction of weights to dropout each epoch

    examples = y.shape[0]
    test_examples = int(examples * test_fraction)
    train_examples = examples - test_examples

    # splitting data into test and train
    x_train = x[0:train_examples:, :, :, :]
    x_test = x[train_examples:examples:, :, :, :]

    # splitting labels into test and train
    y_train = y[0:train_examples:]
    y_test = y[train_examples::]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')

    # build model topology
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(96, 96, 3)  # setting image input shape
        )
    )
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    # get final model test set performance
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
