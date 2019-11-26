import cv2
import numpy as np
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

img_path='res'
CLASS_MAP = { #any new gesture has to be added here
    "hello": 0,
    "none": 1
}

NUM_CLASSES = len(CLASS_MAP)


def mapper(val):
    return CLASS_MAP[val]


def get_model():
    model = Sequential([
        SqueezeNet(input_shape=(227, 227, 3), include_top=False),
        Dropout(0.5),
        Convolution2D(NUM_CLASSES, (1, 1), padding='valid'),
        Activation('relu'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ])
    return model

dataset = []
for directory in os.listdir(img_path):
    path = os.path.join(img_path, directory)
    if not os.path.isdir(path):
        continue
    for item in os.listdir(path):
        # to make sure no hidden files get in our way
        if item.startswith("."):
            continue
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, directory])

data, labels = zip(*dataset)
labels = list(map(mapper, labels))
labels = np_utils.to_categorical(labels) #encoding now
model = get_model()
model.compile(
    optimizer=Adam(lr=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
X=np.array(data)
Y=np.array(labels)
model.fit(X, Y, validation_split=0.33, epochs=10, batch_size=10)
model.save("gesturecheck.h5")