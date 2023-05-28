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


# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

width =   96
height=96

#names in array for the trained object
class_names = [ 'WATERBOTTLE', 'PHONE', 'GLASSES', 'MOUSE']
#initializing the directory
waterbottle = 'img_1/0.png'
phone = 'img_2/0.png'
glasses = 'img_3/0.png'
mouse = 'img_4/0.png'

classes = None
predicted_classes = []
true_labels = []

#arrays for the images
imgs = (waterbottle,phone, glasses, mouse)

#def predict_(img_path):
classes = None
#Live Predictions using camera
CAMERA = cv2.VideoCapture(0)
camera_height = 500

while(True):
  
    _, frame = CAMERA.read()

    frame = cv2.flip (frame, 1)

    #Resacle the images output
    aspect = frame.shape[1]/float(frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame, (res, camera_height))
    x1 = int(frame.shape[1] * 0.25)
    y1 = int(frame.shape[0] * 0.25)
    
    x2 = int(frame.shape[1] * 0.75)
    y2 = int(frame.shape[0] * 0.75)

    roi = frame[y1+2:y2-2, x1+2:x2-2]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (width, height))
    roi_x = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi_x)
    type_1_x, type_2_x, type_3_x, type_4_x = predictions[0]

    #The green rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Predictions / Labels
    type_1_txt = '{} - {}%'.format(class_names[0], int(type_1_x*100))
    cv2.putText(frame, type_1_txt, (70, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_2_txt = '{} - {}%'.format(class_names[1], int(type_2_x*100))
    cv2.putText(frame, type_2_txt, (70, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_3_txt = '{} - {}%'.format(class_names[2], int(type_3_x*100))
    cv2.putText(frame, type_3_txt, (70, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    type_4_txt = '{} - {}%'.format(class_names[3], int(type_4_x*100))
    cv2.putText(frame, type_4_txt, (70, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(240,240,240), 2)

    cv2.imshow("OBJECT DETECTION", frame)

    #Controls q = quit/ s = capturing
    key = cv2.waitKey(1)

    if key & 0xff == ord('q'):
        break
    plt.show()
CAMERA.release()
cv2.destroyAllWindows()