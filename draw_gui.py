import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter('ignore')

import argparse
import cv2
import numpy as np
import pickle
import tkinter as tk

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from keras.models import model_from_yaml
from scipy.misc import imread, imresize

def init_gui(model, width=400, height=400):
    def exit():
        frame.destroy()

    def paint(event, pen_size=10):
    	x1, y1 = (event.x - pen_size), (event.y - pen_size)
    	x2, y2 = (event.x + pen_size), (event.y + pen_size)
    	window.create_oval(x1, y1, x2, y2, fill='black', outline='black')
    	draw.ellipse([x1, y1, x2, y2], (0, 0, 0), outline=(0, 0, 0))

    def clear():
        window.delete('all')

    def predict():
        image1 = image.crop(image.getbbox())
        plt.imshow(image1)
        plt.show()
        label.configure(text='Prediction: A')

    frame = tk.Tk()
    frame.title('Draw Character')

    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    window = tk.Canvas(frame, width=width, height=height + 20)
    button_predict = tk.Button(frame, text='Predict', command=predict)
    button_clear = tk.Button(frame, text='Clear', command=clear)
    label = tk.Label(frame, text='Prediction: ')

    window.bind('<B1-Motion>', paint)
    window.configure(background='white')
    window.pack(expand=True, fill='both')
    label.pack(expand=True)
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
    init_gui(model)
