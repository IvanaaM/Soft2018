import cv2
import numpy as np
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate
from math import hypot

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

from keras.datasets import mnist

def create_ann():
    
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    print(ann.summary())

    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def convert_output(y_train):
    nn_outputs = []
    # i - redni broj
    # j - vrednost
    for i, j in enumerate(y_train):
        output = np.zeros(10)
        output[j] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        width, height = region.shape
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

        
    return ready_for_ann

def matrix_to_vector(image):
    return image.flatten()

def scale_to_range(image): 
    
    return image/255

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_for_train = prepare_for_ann(x_train[:20000])
y_for_train = convert_output(y_train[:20000])

ann = create_ann()
ann = train_ann(ann, x_for_train, y_for_train)

ann.save('modelSa30.h5')
