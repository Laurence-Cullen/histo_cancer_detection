import os
import pathlib

import numpy as np
import pandas as pd
import psutil
from PIL import Image

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout

train_directory = pathlib.Path('train')
image_extension = 'tif'


def get_train_image(name):
    img = Image.open(train_directory / (name + '.' + image_extension))
    img.load()
    return np.asarray(img, dtype="int32")


def load_data(file_names, max_files=30000):
    counter = 0

    print_every = 100
    stack_every = 10000

    arrays = []
    data = None

    for file_name in file_names:

        if counter % print_every == 0:
            print(f'loaded {counter} files')
            print("memory usage:", psutil.virtual_memory())
            print("swap memory usage:", psutil.swap_memory())

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

    return np.stack([data] + arrays)


def normalize_image_data(data):
    return (data / 255.0) - 0.5


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def main():
    labels = pd.read_csv('train_labels.csv')

    y = labels['label'].values
    file_names = labels['id'].values

    x = load_data(file_names)
    x = normalize_image_data(x)

    # trim labels to the same size as the number of loaded in files
    y = y[0:x.shape[0]:]

    y, x = unison_shuffled_copies(y, x)

    test_fraction = 0.1
    batch_size = 128
    num_classes = 1
    epochs = 200

    examples = y.shape[0]
    test_examples = int(examples * test_fraction)
    train_examples = examples - test_examples

    x_train = x[0:train_examples:, :, :, :]
    x_test = x[train_examples:examples:, :, :, :]

    y_train = y[0:train_examples:]
    y_test = y[train_examples::]

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    print('x_test shape:', x_test.shape)
    print(x_test.shape[0], 'test samples')

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(5, 5),
            activation='relu',
            input_shape=(96, 96, 3)
        )
    )
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    main()
