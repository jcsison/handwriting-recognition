import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import cv2
import numpy as np
import pickle

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
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
    def process_image(image):
        def background_color(img):
            (values, counts) = np.unique(img, return_counts=True)
            return values[np.argmax(counts)]

        def crop(img, color=0, padding=2):
            mask = img != color
            if True not in mask:
                return img
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0) - padding
            x1, y1 = coords.max(axis=0) + 1 + padding
            x0, y0 = x0 * (x0 > 0), y0 * (y0 > 0)
            x1, y1 = x1 * (x1 > 0), y1 * (y1 > 0)
            return img[x0:x1, y0:y1]

        def square(img, color=0):
            (x, y) = img.shape
            if x > y:
                padding = ((0, 0), ((x - y) // 2, (x - y) // 2))
            else:
                padding = (((y - x) // 2, (y - x) // 2), (0, 0))
            return np.pad(img, padding, mode='constant', constant_values=color)

        img_c = image.crop(image.getbbox()).convert('L')
        img_c = ImageOps.invert(img_c)
        w, h = img_c.size
        w, h = w // 2, h // 2
        x, y = ndimage.measurements.center_of_mass(np.asarray(img_c))
        img_t = img_c.transform(img_c.size, Image.AFFINE, (1, 0, y - h, 0, 1, x - w), fill=0)
        img_t = np.asarray(img_t.convert('L'))
        ret, img = cv2.threshold(img_t, 127, 255, cv2.THRESH_BINARY)
        if background_color(img) != 0:
            img = np.invert(img)
        img = cv2.GaussianBlur(img, (1, 1), 0)
        img = crop(img)
        img = square(img)
        return img

    img = imread(file_path, mode='L')
    img = process_image(Image.fromarray(img))
    img = imresize(img, (28, 28))
    img = ndimage.rotate(img, 90)
    img = cv2.flip(img, 0)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    out = model.predict(img)
    print('Prediction:', chr(mapping[(int(np.argmax(out, axis=1)[0]))]))
    print('Confidence:', str(max(out[0]) * 100)[:6])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='python3 predict.py -f [file_path]')
    parser.add_argument('-f', '--file', type=str, help='Image file path', required=True)
    args = parser.parse_args()

    model = load_model()
    mapping = pickle.load(open('model/mapping.p', 'rb'))
    predict(args.file, mapping)
