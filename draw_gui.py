import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import cv2
import numpy as np
import pickle
import tkinter as tk

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from keras.models import model_from_yaml
from scipy.misc import imresize

def init_gui(model, mapping, width=400, height=400):
    def exit():
        frame.destroy()

    def paint(event, pen_size=10):
    	x1, y1 = (event.x - pen_size), (event.y - pen_size)
    	x2, y2 = (event.x + pen_size), (event.y + pen_size)
    	window.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    	draw.ellipse([x1, y1, x2, y2], (0, 0, 0), outline=(0, 0, 0))

    def clear():
        image.paste(Image.new('RGB', (width, height), (255, 255, 255)))
        window.delete('all')
        label1_var.set('Prediction: ')
        label2_var.set('Confidence: ')

    def predict():
        def crop(img, color=0):
            mask = img != color
            coords = np.argwhere(mask)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1
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
        img = crop(img)
        img = square(img)
        img = imresize(img, (28, 28))
        plt.imshow(img)
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32') / 255

        out = model.predict(img)
        prediction = chr(mapping[(int(np.argmax(out, axis=1)[0]))])
        confidence = max(out[0]) * 100
        label1_var.set('Prediction: ' + prediction)
        label2_var.set('Confidence: {0:.1f}'.format(confidence))
        plt.show()

    frame = tk.Tk()
    frame.title('Draw Handwritten Character')

    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    window = tk.Canvas(frame, width=width, height=height + 20)
    button_predict = tk.Button(frame, text='Predict', command=predict)
    button_clear = tk.Button(frame, text='Clear', command=clear)

    label1_var = tk.StringVar()
    label1_var.set('Prediction: ')
    label1 = tk.Label(frame, textvariable=label1_var)
    label2_var = tk.StringVar()
    label2_var.set('Confidence: ')
    label2 = tk.Label(frame, textvariable=label2_var)


    window.bind('<B1-Motion>', paint)
    window.configure(background='white')
    window.pack(expand=True, fill='both')
    label1.pack(expand=True)
    label2.pack(expand=True)
    button_predict.pack(expand=True, fill='x', side='left')
    button_clear.pack(expand=True, fill='x', side='right')

    frame.protocol('WM_DELETE_WINDOW', exit)
    frame.mainloop()

def load_model():
    yaml_file = open('model/model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()

    model = model_from_yaml(loaded_model_yaml)
    model.load_weights('model/model.h5')
    return model

if __name__ == '__main__':
    model = load_model()
    mapping = pickle.load(open('model/mapping.p', 'rb'))
    init_gui(model, mapping)
