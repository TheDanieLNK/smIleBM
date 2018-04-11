# -*- coding: utf-8 -*-

# this code was built using Google Colab
# author: Daniel Nkemelu
# date: April 11, 2018

# import modules
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import imutils

# create model architecture: LeNet
class MyModel:
  @staticmethod
  def build(width, height, depth, classes):
  # initialize the model
    model = Sequential()
    inputShape = (height, width, depth)

    # if we are using "channels first", we update the input shape
    if K.image_data_format() == "channels_first":
      inputShape = (depth, height, width)
      
    # set up our architecture
    model.add(Conv2D(20, (5, 5), padding="same",
    input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return our model architecture
    return model

# load datasets
train_data_dir_pos = 'smiles/positives/positives7/'
train_data_dir_neg = 'smiles/negatives/negatives7/'

# initializing inputs and labels lists
inputs = []
labels = []

# insert positive image into input array
for item in os.listdir(train_data_dir_pos):
  pos_image = cv2.imread('SMILEs/positives/positives7/'+item, 0)
  pos_image = imutils.resize(pos_image, width=28)
  pos_image = img_to_array(pos_image)
  inputs.append(pos_image)
  labels.append("smiling")

# insert negative image into input array
for itemn in os.listdir(train_data_dir_neg):
  pos_imagen = cv2.imread('SMILEs/negatives/negatives7/'+itemn, 0)
  try:
    pos_imagen = imutils.resize(pos_imagen, width=28)
    pos_imagen = img_to_array(pos_imagen)
    inputs.append(pos_imagen)
    labels.append("not-smiling")
  except AttributeError:
    continue

# normalise input by scaling to [0,1]
inputs = np.array(inputs, dtype="float") / 255.0
labels = np.array(labels)

# converting the labels to one-hot encoding
lenc = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(lenc.transform(labels), 2)

# account for imbalance for 
classNum = labels.sum(axis=0)
classWeight = classNum.max() / classNum

# divide our data into a training and validation set
(trainX, testX, trainY, testY) = train_test_split(inputs, labels, 
                              test_size=0.20, stratify=labels, random_state=26)

# build the model using adam optimizer
print("[Building model...]")
model = MyModel.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", 
              optimizer="adam", metrics=["accuracy"])

# train the network
print("[Training network...]")
mdl = model.fit(trainX, trainY, validation_data=(testX, testY), 
                 class_weight=classWeight, batch_size=64, epochs=14, verbose=1)

# evaluate the network
print("Evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=lenc.classes_))

# save the model to disk
print("[Save model...]")
model.save_weights("model_weights.h5")

# save the json version of our model for the flask frontend
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# download model and model json from colab
from google.colab import files
files.download('model_weights.h5')
files.download('model.json')

# plot a visualization of model performance
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 16), mdl.history["loss"], label="Train Loss")
plt.plot(np.arange(0, 16), mdl.history["val_loss"], label="Val Loss")
plt.plot(np.arange(0, 16), mdl.history["acc"], label="Acc")
plt.plot(np.arange(0, 16), mdl.history["val_acc"], label="Val Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
