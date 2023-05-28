
#libraries needed for the program
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



#names in array for the trained object
class_names = [ 'WATERBOTTLE', 'PHONE', 'GLASSES', 'MOUSE']

#start for the live camera
CAMERA = cv2.VideoCapture (0)
camera_height = 500


raw_frames_type_1 = []
raw_frames_type_2 = []
raw_frames_type_3 = []
raw_frames_type_4 = []


while CAMERA.isOpened ():
    # Read a new camera frame 
    _, frame = CAMERA.read()

    frame = cv2.flip (frame, 1)
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect* camera_height)
    frame = cv2.resize(frame,(res,camera_height))
    
    x1 = int(frame.shape[1] * 0.25)
    y1 = int(frame.shape[0] * 0.25)
    
    x2 = int(frame.shape[1] * 0.75)
    y2 = int(frame.shape[0] * 0.75)

     # The green rectangle 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow ("Capturing", frame)
    key = cv2.waitKey (1)

#condition for the live camera/prompts the user
    if key & 0xff == ord ('q'):
        break
    
    elif key & 0xFF == ord ('1'):
        raw_frames_type_1.append (frame)
        
    elif key & 0xFF == ord ('2'):
        raw_frames_type_2.append (frame)
        
    elif key & 0xFF == ord ('3'):
        raw_frames_type_3.append (frame)
        
    elif key & 0xFF == ord ('4'):
        raw_frames_type_4.append (frame)

CAMERA.release()
cv2.destroyAllWindows()



save_width = 339
save_height = 400

retval = os.getcwd()
print ("Current working directopry %s" % retval)

print ('img_1: ', len (raw_frames_type_1))
print ('img_2: ', len (raw_frames_type_2))
print ('img_3: ', len (raw_frames_type_3))
print ('img_4: ', len (raw_frames_type_4))


#cropping the captured images

for i, frame in enumerate(raw_frames_type_1):
    # Get ROI
    frame_height, frame_width, _ = frame.shape
    roi_left = int((frame_width / 2) - (save_width / 2))
    roi_right = int((frame_width / 2) + (save_width / 2))
    roi_top = int((frame_height / 2) - (save_height / 2))
    roi_bottom = int((frame_height / 2) + (save_height / 2))
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Parse BGR to RGB
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to desired dimensions
    roi = cv2.resize(roi, (save_width, save_height))

    # Save the cropped and resized image
    cv2.imwrite('img_1/{}.png'.format(i), roi)

for i, frame in enumerate (raw_frames_type_2):

    frame_height, frame_width, _ = frame.shape
    roi_left = int((frame_width / 2) - (save_width / 2))
    roi_right = int((frame_width / 2) + (save_width / 2))
    roi_top = int((frame_height / 2) - (save_height / 2))
    roi_bottom = int((frame_height / 2) + (save_height / 2))
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Parse BGR to RGB
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to desired dimensions
    roi = cv2.resize(roi, (save_width, save_height))

    # Save the cropped and resized image
    cv2.imwrite('img_2/{}.png'.format(i), roi)

for i, frame in enumerate (raw_frames_type_3):

    frame_height, frame_width, _ = frame.shape
    roi_left = int((frame_width / 2) - (save_width / 2))
    roi_right = int((frame_width / 2) + (save_width / 2))
    roi_top = int((frame_height / 2) - (save_height / 2))
    roi_bottom = int((frame_height / 2) + (save_height / 2))
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Parse BGR to RGB
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to desired dimensions
    roi = cv2.resize(roi, (save_width, save_height))

    # Save the cropped and resized image
    cv2.imwrite('img_3/{}.png'.format(i), roi)

for i, frame in enumerate (raw_frames_type_4):

    #Get roi
    frame_height, frame_width, _ = frame.shape
    roi_left = int((frame_width / 2) - (save_width / 2))
    roi_right = int((frame_width / 2) + (save_width / 2))
    roi_top = int((frame_height / 2) - (save_height / 2))
    roi_bottom = int((frame_height / 2) + (save_height / 2))
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]


    # Parse BGR to RGB
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to desired dimensions
    roi = cv2.resize(roi, (save_width, save_height))

    # Save the cropped and resized image
    cv2.imwrite('img_4/{}.png'.format(i), roi)



width = 150
height = 150

#arrays created to put the captured images
images_type_1 = []
images_type_2 = []
images_type_3 = []
images_type_4 = []



#load the images from the directory
    #used tf.keras.utils.load_img for the updated version of tensorflow/keras
for image_path in glob.glob ('img_1/*.*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)

    images_type_1.append(x)

for image_path in glob.glob('img_2/*.*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)

    images_type_2.append(x)
 
for image_path in glob.glob('img_3/*.*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)

    images_type_3.append(x)
 
for image_path in glob.glob('img_4/*.*'):
    image = tf.keras.utils.load_img(image_path, target_size=(width, height))
    x = tf.keras.utils.img_to_array(image)

    images_type_4.append(x)

plt.figure(figsize=(12,8))



#show in the figure the captured images
for i,x in enumerate(images_type_1[:5]):
    plt.subplot(1, 5, i+1)
    image =tf.keras.utils.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{}'.format(class_names[0]))

plt.show()
plt.figure(figsize=(12,8))


for i,x in enumerate(images_type_2[:5]):
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)   

    plt.axis('off')
    plt.title('{}'.format(class_names[1]))

plt.show()
plt.figure(figsize=(12,8))

for i,x in enumerate(images_type_3[:5]):
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image) 

    plt.axis('off')
    plt.title('{}'.format(class_names[2]))

plt.show()
plt.figure(figsize=(12,8))

for i,x in enumerate(images_type_4[:5]):
    plt.subplot(1, 5, i+1)
    image = tf.keras.utils.array_to_img(x)
    plt.imshow(image)

    plt.axis('off')
    plt.title('{}'.format(class_names[3]))

plt.show()
plt.figure(figsize=(12,8))


# Prepare Image to Tensor
X_type_1 = np.array(images_type_1)
X_type_2 = np.array(images_type_2)
X_type_3 = np.array(images_type_3)
X_type_4 = np.array(images_type_4)


#Check the image shape using .shape()
print (X_type_1.shape)
print (X_type_2.shape)
print (X_type_3.shape)
print (X_type_4.shape)

#print(X_type_2)

 
X = np.concatenate((X_type_1, X_type_2), axis=0)

if len (X_type_3):
    X = np.concatenate((X, X_type_3), axis=0)
    
if len (X_type_4):
    X = np.concatenate((X, X_type_4), axis=0)

#Scaling the data to 1-0
X = X / 255.0

print(X.shape)



y_type_1 = [0 for item in enumerate (X_type_1)]
y_type_2 = [1 for item in enumerate (X_type_2)]
y_type_3 = [2 for item in enumerate (X_type_3)]
y_type_4 = [3 for item in enumerate (X_type_4)]

y = np.concatenate((y_type_1,y_type_2),axis=0)

if len (y_type_3):
    y = np.concatenate((y,y_type_3),axis=0)
    
if len (y_type_4):
    y = np.concatenate((y,y_type_4),axis=0)
    
y = to_categorical(y,num_classes=len(class_names))

print(y.shape)


#default Parameters
conv_1 = 16
conv_1_drop = 0.2

conv_2 = 32
conv_2_drop = 0.2

dense_1_n = 1024
dense_1_drop = 0.2

dense_2_n =512
dense_2_drop = 0.2

lr = 0.001
epochs = 15
batch_size =10
color_channels =  3

#building the model
def build_model (conv_1_drop = conv_1_drop, conv_2_drop = conv_2_drop, 
                 dense_1_n = dense_1_n, dense_1_drop = dense_1_drop,
                 dense_2_n = dense_2_n, dense_2_drop = dense_2_drop,
                 lr = lr):
    
    model = Sequential()
    
    model.add(Convolution2D(conv_1,(3,3),input_shape = (width,height, color_channels),
                            activation = 'relu'))
    
    model.add (MaxPooling2D(pool_size=(2,2)))
    
    model.add (Dropout(conv_1_drop))
    

    model.add(Convolution2D(conv_2,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add (Dropout(conv_1_drop))
    
    
    model.add(Flatten())
    
    model.add(Dense(dense_1_n , activation='relu'))
    model.add(Dropout(dense_1_drop))
    
    
    model.add(Dense(dense_2_n , activation='relu'))
    model.add(Dropout(dense_2_drop))
    
    model.add (Dense(len(class_names),activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer = Adam(clipvalue=0.5),
                  metrics=['accuracy'])
    
    
    return model

model = build_model()

model.summary()
model.save('my_model.h5')

np.set_printoptions(suppress=True, floatmode='fixed')
history = model.fit(X,y, validation_split=0.10,epochs=15,batch_size=5)

print(history)

#model evaluation
scores = model.evaluate(X, y, verbose=0)

#printing accuracy for the trained object
print("Accuracy: %.2f%%" % (scores[1]*100))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss and accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#Prediction
def plt_show(img):
    plt.imshow(img)
    plt.show()
    
    
#initializing the directory
waterbottle = 'img_1/0.png'
phone = 'img_2/0.png'
glasses = 'img_3/0.png'
mouse = 'img_4/0.png'

#arrays for the images
imgs = (waterbottle,phone, glasses, mouse)

#def predict_(img_path):
classes = None
predicted_classes = []
true_labels = []

#tf.keras.utils.load_img used for the new version of Tensorflow
for i in range(len( imgs)):
    type_ = tf.keras.utils.load_img(imgs[i], target_size=(width, height))
    plt.imshow(type_) # type: ignore
    plt.show()

    type_x = np.expand_dims(type_, axis=0) # type: ignore
    prediction = model.predict(type_x)
    index = np.argmax(prediction)
    print(class_names[index])
    classes = class_names[index]
    predicted_classes.append(class_names[index])
    true_labels.append(class_names[i % len (class_names)])
#parameters are set to display the class names along the x-axis and y-axis, respectively
cm = confusion_matrix(class_names, predicted_classes)
f = sns.heatmap (cm, xticklabels= class_names, yticklabels= predicted_classes, annot= True)






                  
                  
