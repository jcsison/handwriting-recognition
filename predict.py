import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import cv2
import numpy as np
import pickle

from matplotlib import pyplot as plt
from keras.models import model_from_yaml
from scipy.misc import imread, imresize

def load_model():
    yaml_file = open('model/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('model/model.h5')
    return model

def predict(file_path, mapping):
    def square(img, color):
        (x, y) = img.shape
        if x > y:
            padding = ((0, 0), ((x - y) // 2, (x - y) // 2))
        else:
            padding = (((y - x) // 2, (y - x) // 2), (0, 0))
        return np.pad(img, padding, mode='constant', constant_values=color)

    def background_color(img):
        (values, counts) = np.unique(img, return_counts=True)
        return values[np.argmax(counts)]

    img = imread(file_path, mode='L')
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    color = background_color(img)
    img = square(img, color)
    if color != 0:
        img = np.invert(img)
    img = imresize(img, (28, 28))
    plt.imshow(img)
    plt.show()
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255

    out = model.predict(img)

    print('Prediction:', chr(mapping[(int(np.argmax(out, axis=1)[0]))]))
    print('Confidence:', str(max(out[0]) * 100)[:6])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python3 predict.py -f [file_name]')
    parser.add_argument('-f', '--file', type=str, help='Image file path', required=True)
    args = parser.parse_args()

    model = load_model()
    mapping = pickle.load(open('model/mapping.p', 'rb'))
    predict(args.file, mapping)
