# handwriting-recognition
Implementation of handwriting recognition using machine learning.

## Requirements
- [Python 3.6](https://www.python.org/downloads/)
- [TensorFlow 1.6](https://www.tensorflow.org/install/)
- [SciPy 1.0](https://scipy.org/install.html)
- [Keras 2.1](https://keras.io/#installation)
- [EMNIST dataset](https://www.nist.gov/itl/iad/image-group/emnist-dataset)

## Instructions
### Training
Before training, ensure that `emnist-balanced.mat` from the [EMNIST dataset](https://cloudstor.aarnet.edu.au/plus/index.php/s/7YXcasTXp727EqB/download) is stored within the `data/` directory.

Run the training script using:

``` bash
python3 train.py
```

A model h5, YAML, and Pickle file will be generated within the `model/` directory.

### Prediction
Once the model has been created, a single handwritten character can be predicted by running:

``` bash
python3 predict.py [file1] ...
```

Note: multiple files can be predicted in one instance and a wildcard `*` can be used to process all files within a directory.

### Prediction GUI
A Tkinter GUI has been created to test the model implementation more easily. This can be started with the command:

``` bash
python3 draw_gui.py
```

### To-do
- ~~GUI testing environment for live demo~~
- ~~Train on byclass dataset (currently using balanced)~~
- ~~Improve prediction accuracy (should improve after switching to byclass)~~
- Compare byclass with other EMNIST datasets
- Improve image processing
- Train using other learning algorithms (if time permits)
    - SVM
    - Random forest
- Image process strings of letters (if time permits)
