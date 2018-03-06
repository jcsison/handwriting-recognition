import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import keras
import numpy as np
import pickle

from keras.layers import Conv3D, MaxPooling2D, Convolution2D, Dropout, Dense, Flatten, LSTM
from keras.models import Sequential, save_model
from keras.utils import np_utils
from scipy.io import loadmat
from tensorflow.python.lib.io import file_io

def load_data(mat_file_path, job_path, width=28, height=28):
    def rotate(img):
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    if not os.path.exists(job_path):
        os.makedirs(job_path)

    file_stream = file_io.FileIO(mat_file_path, mode='r')
    mat = loadmat(file_stream)
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    with file_io.FileIO(job_path + '/mapping.p', mode='w') as output_f:
        pickle.dump(mapping, output_f)

    size = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0].reshape(size, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1]

    size = len(mat['dataset'][0][0][1][0][0][0])
    testing_images = mat['dataset'][0][0][1][0][0][0].reshape(size, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1]

    length = len(testing_images)
    for i in range(len(testing_images)):
        testing_images[i] = rotate(testing_images[i])

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

def train(job_path, model, training_data, batch_size=256, epochs=10):
    (x_train, y_train), (x_test, y_test), mapping, nb_classes = training_data

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    tb_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(job_path, 'Graph/'),
        histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
        validation_data=(x_test, y_test), callbacks=[tb_callback])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model_yaml = model.to_yaml()
    with file_io.FileIO(job_path + '/model.yaml', mode='w') as output_f:
        output_f.write(model_yaml)
    model.save('model.h5')
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_path + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python3 train_gcloud.py -t [file path] -j [job directory]')
    parser.add_argument('-t', '--train-file', type=str, help='Training dataset file path',
        dest='train', required=True)
    parser.add_argument('-j', '--job-dir', type=str, help='Job directory', dest='job', required=True)
    args = parser.parse_args()

    np.random.seed(10)
    training_data = load_data(args.train, args.job)
    model = build_net(training_data)
    train(args.job, model, training_data)
