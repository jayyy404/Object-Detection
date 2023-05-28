# Object-Detection

#install the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Activation , Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import os
from glob import glob
import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.applications import inception_v3
import time
from keras import preprocessing
from keras_preprocessing import image


#create folders in the same directory named as {"img_1, img_2, img_3, img_4"}
