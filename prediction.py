import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import keras
import numpy as np
from tqdm import tqdm
import warnings
import gc

ROWS = 224
COLS = 224
CHANNELS = 3


def data_loading(limit):
    TEST_DIR = os.getcwd()
    test_images = [os.path.join(TEST_DIR, i) for i in (os.listdir(TEST_DIR)[:limit])]
    return test_images

import os

import cv2
import keras
import numpy as np
from tqdm import tqdm

ROWS = 224
COLS = 224
CHANNELS = 3


def data_loading(limit):
    TEST_DIR = os.getcwd()
    test_images = []
    dirs = os.listdir(TEST_DIR)[:limit]
    if 'cats' in dirs: dirs.remove('cats')
    if 'dogs' in dirs: dirs.remove('dogs')
    if 'model_trained' in dirs: dirs.remove('model_trained')
    if 'prediction.py' in dirs: dirs.remove('prediction.py')
    test_images = [os.path.join(TEST_DIR, i) for i in dirs]
    return np.array(test_images)


def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        print('Wrong path:', file_path)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def process_images(images):
    data = []
    print("\n\nreading next batch : ", len(images))
    for i in tqdm(images):
        image = read_image(i)
        data.append(image)
    return np.array(data)


def start_prediction():
    model = keras.models.load_model("model_trained")
    cwd = os.getcwd()
    while len(os.listdir(cwd)) > 4:
        test = process_images(data_loading(1000))
        predictions = model.predict(test)
        cwd = os.getcwd()
        path = os.path.join(cwd, "dogs")
        if not os.path.isdir(path):
            os.makedirs(path)
        path = os.path.join(cwd, "cats")
        if not os.path.isdir(path):
            os.makedirs(path)
        images = os.listdir(cwd)
        print("processing batch : ", len(test))
        for i in tqdm(range(0, len(test))):
            if predictions[i, 0] >= 0.5:
                os.rename(os.path.join(cwd, images[i]), os.path.join(os.path.join(cwd, "dogs"), images[i]))
            else:
                os.rename(os.path.join(cwd, images[i]), os.path.join(os.path.join(cwd, "cats"), images[i]))
        del test
        del images
        del predictions
        gc.collect()

warnings.filterwarnings("ignore")
start_prediction()