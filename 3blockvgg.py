### Load the data ###

labels = ['normal','abnormal']
image_size = 224

# Borrowed "get_data" function from
# https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/

import os 
import cv2 
import numpy as np

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir,label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path,img))[...,::-1] # Convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (image_size, image_size)) # Reshape images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

dataset = get_data('../two_class_post_weld') ## define current path to the original dataset folder

from sklearn.model_selection import train_test_split
train, val = train_test_split(dataset, test_size=1/3)

### Compare the number of the images in both cases ###

import seaborn as sns

count_number = []
for i in train:
    if(i[1]==0):
        count_number.append("normal")
    else:
        count_number.append("abnormal")

sns.set_style('whitegrid')
sns.countplot(count_number)

### Visualize a random image from both classes ###

import matplotlib.pyplot as plt

# visualize a normal welding image
plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

# visualize a abnormal welding image
plt.figure(figsize = (5,5))
plt.imshow(train[-1][0])
plt.title(labels[train[-1][1]])

### Data Preprocessing ###

# split the data into feature and label

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# normalize the data

import numpy as np

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, image_size, image_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, image_size, image_size,1)
y_val = np.array(y_val)

# data augmentation 

from keras.preprocessing.image import ImageDataGenerator

data_generator = ImageDataGenerator(
    featurewise_center=False, # False to input mean zero
    samplewise_center=False, # False tosample mean zero
    featurewise_std_normalization=False, # False to divide input by its standard deviation
    samplewise_std_normalization=False, # False to divide sample by its standard deviation
    zca_whitening=False, # False to apply ZCA whitening
    rotation_range=30, # rotate images in 25 degree
    zoom_range=0.2, # zoom images 0.3 times
    width_shift_range=0.1, # shift images 0.1 horizontally (fraction of total width)
    height_shift_range=0.1, # shift images 0.1 vertically (fraction of total height)
    horizontal_flip=True, # flip images horizontally
    vertical_flip=False) # flip images vertically

data_generator.fit(x_train)

### Build the model ###

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout

model = Sequential()
model.add(Conv2D(64,(3,3),padding="same",activation="relu",input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Conv2D(256,(3,3),padding="same",activation="relu"))
model.add(Conv2D(256,(3,3),padding="same",activation="relu"))
model.add(Conv2D(256,(3,3),padding="same",activation="relu"))
model.add(Conv2D(256,(3,3),padding="same",activation="relu"))
model.add(MaxPool2D((2,2),strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2,activation="softmax"))

model.summary()

### Evaluate the results ###

import tensorflow as tf
from keras.optimizers import SGD, Adam

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=500, validation_data=(x_val,y_val))

acc= history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['normal welding (Class 0)','abnormal welding(Class 1)']))
