import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

from keras.models import model_from_yaml
from scipy.misc import imread, imresize

from matplotlib import pyplot as plt

import argparse
import numpy as np
import pickle

def load_model():
    yaml_file = open('model/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('model/model.h5')
    return model

def predict(file_path, mapping):
    def crop(img, crop):
        y, x = img.shape
        x_c = (x // 2) - (crop // 2)
        y_c = (y // 2) - (crop // 2)
        return img[y_c:y_c + crop, x_c:x_c + crop]

    img = imread(file_path, mode='L')
    # img = crop(img, min(img.shape[0], img.shape[1]))
    img = np.invert(img)
    img = imresize(img, (28, 28))
    # plt.imshow(img)
    # plt.show()
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
