import os
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

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
    def show_compare(image1, image2):
        plt.subplot(121)
        plt.imshow(image1, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122)
        plt.imshow(image2, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def process_image(image):
        def background_color(img):
            (values, counts) = np.unique(img, return_counts=True)
            return values[np.argmax(counts)]

        def crop(img, color=0):
            padding = max(img.shape) // 40
            mask = img != color
            if True not in mask:
                return img
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
            x0, y0 = x0 * (x0 > 0) - padding, y0 * (y0 > 0) - padding
            x1, y1 = x1 * (x1 > 0) + padding, y1 * (y1 > 0) + padding
            x0 = max(0, x0)
            y0 = max(0, y0)
            return img[x0:x1, y0:y1]

        def square(img, color=0):
            (x, y) = img.shape
            if x > y:
                padding = ((0, 0), ((x - y) // 2, (x - y) // 2))
            else:
                padding = (((y - x) // 2, (y - x) // 2), (0, 0))
            return np.pad(img, padding, mode='constant', constant_values=color)

        img = image.crop(image.getbbox()).convert('L')
        img = np.asarray(img)
        img = cv2.medianBlur(img, 5)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.bilateralFilter(img, 7, 100, 100)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if background_color(img) != 0:
            img = np.invert(img)
        img = crop(img)
        img = square(img)
        return img

    image = imread(file_path)
    img = imread(file_path, mode='L')
    img = process_image(Image.fromarray(img))
    img = imresize(img, (28, 28))
    img_ref = img
    img = ndimage.rotate(img, 90)
    img = cv2.flip(img, 0)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255
    out = model.predict(img)
    print('Prediction:', chr(mapping[(int(np.argmax(out, axis=1)[0]))]))
    print('Confidence:', str(max(out[0]) * 100)[:6])
    show_compare(image, img_ref)
    return max(out[0]) * 100

if __name__ == '__main__':
    model = load_model()
    mapping = pickle.load(open('model/mapping.p', 'rb'))
    if len(sys.argv) > 2:
        confidence = 0
        for i in sys.argv[1:]:
            confidence += predict(i, mapping)
        confidence /= len(sys.argv[1:])
        print('Average Confidence:', str(confidence)[:6])
    elif len(sys.argv) > 1:
        predict(sys.argv[1], mapping)
    else:
        print("predict.py: missing file operand")
        print("usage: python3 predict.py [file1] ...")
