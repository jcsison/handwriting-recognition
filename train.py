import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import keras
import numpy as np
import pickle

from keras.layers import Conv3D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat

def load_data(mat_file_path, width=28, height=28):
    def rotate(img):
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    if not os.path.exists('model/'):
        os.makedirs('model/')

    mat = loadmat(mat_file_path)
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('model/mapping.p', 'wb'))

    size = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0].reshape(size, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1]

    size = len(mat['dataset'][0][0][1][0][0][0])
    testing_images = mat['dataset'][0][0][1][0][0][0].reshape(size, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1]

    length = len(testing_images)
    for i in range(len(testing_images)):
        print('%d/%d (%.2lf%%)' % (i + 1, length, ((i + 1) / length) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    print()

    training_images = training_images.astype('float32') / 255
    testing_images = testing_images.astype('float32') / 255

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

def build_net(training_data, width=28, height=28):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data
    input_shape = (height, width, 1)

    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size, padding='valid',
        input_shape=input_shape, activation='relu'))
    model.add(Convolution2D(nb_filters, kernel_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    print(model.summary())

    return model

def train(model, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    tb_callback = keras.callbacks.TensorBoard(log_dir='./model/Graph',
        histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[tb_callback])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model_yaml = model.to_yaml()
    with open("model/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    save_model(model, 'model/model.h5')

if __name__ == '__main__':
    np.random.seed(10)
    # training_data = load_data('data/emnist-byclass.mat')
    training_data = load_data('data/emnist-balanced.mat')
    model = build_net(training_data)
    train(model, training_data)
