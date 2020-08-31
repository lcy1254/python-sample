#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries 
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import time
import random
import seaborn as sns
import keras
from keras.preprocessing.image import ImageDataGenerator
import math
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard 
import datetime
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from sklearn.model_selection import train_test_split


# In[2]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[3]:


#directories to images
train_dir = "/Users/ChaeyoungLee/Downloads/DATA_CHEST_X_RAY/train"
test_dir = "/Users/ChaeyoungLee/Downloads/DATA_CHEST_X_RAY/test"
val_dir = "/Users/ChaeyoungLee/Downloads/DATA_CHEST_X_RAY/val"
directories = [train_dir, test_dir, val_dir]
categories = ["NORMAL", "PNEUMONIA"]

train_data = []
test_data = []
val_data = []

img_size = 150 #try different img_size and see which one works best 

for directory in directories: 
    for category in categories:
        path = os.path.join(directory, category)
        label_num = int(categories.index(category))
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            res_array = cv2.resize(img_array, (img_size, img_size))
            if directory == train_dir: 
                train_data.append([res_array, label_num])
            elif directory == test_dir:
                test_data.append([res_array, label_num])
            elif directory == val_dir:
                val_data.append([res_array, label_num])


# In[4]:


plt.imshow(train_data[0][0], cmap = 'gray')
print("0 is normal, 1 is pneumonia. This is: " + str(train_data[0][1]))


# In[5]:


train_data = np.array(train_data)
test_data = np.array(test_data)
val_data = np.array(val_data)


# In[6]:


print(train_data.shape, val_data.shape, test_data.shape)
print(train_data[0]) 


# In[7]:


#check data distribution
plt.figure(1, figsize = (15, 5))
n = 0
for r, j in zip([train_data, test_data, val_data], ["train", "test", "val"]):
    n += 1
    l = []
    for i in range(r.shape[0]):
        if r[i][1] == 1:
            l.append("Pneumonia")
        else:
            l.append("Normal")
    plt.subplot(1, 3, n)
    sns.countplot(l)
    plt.title(j)
plt.show()


# In[8]:


#add more to val_data
temp_train_data = train_data
train_data, temp_val_data = train_test_split(temp_train_data, test_size = 0.007, random_state = 10)


# In[9]:


val_data = np.append(val_data, temp_val_data, axis = 0)


# In[10]:


print(val_data.shape)


# In[11]:


#check data distribution again
plt.figure(1, figsize = (15, 5))
n = 0
for r, j in zip([train_data, test_data, val_data], ["train", "test", "val"]):
    n += 1
    l = []
    for i in range(r.shape[0]):
        if r[i][1] == 1:
            l.append("Pneumonia")
        else:
            l.append("Normal")
    plt.subplot(1, 3, n)
    sns.countplot(l)
    plt.title(j)
plt.show()


# In[12]:


#divide into X & Y
train_X = []
train_Y = []
test_X = []
test_Y = []
val_X = []
val_Y = []
#data_sets = [train_X, train_Y, test_X, test_Y, val_X, val_Y]

for features, label in train_data:
    train_X.append(features)
    train_Y.append(label)
    
train_X = np.array(train_X)
train_Y = np.array(train_Y)
    
for features, label in test_data:
    test_X.append(features)
    test_Y.append(label)

test_X = np.array(test_X)/255.0
test_Y = np.array(test_Y)
    
for features, label in val_data:
    val_X.append(features)
    val_Y.append(label)
    
val_X = np.array(val_X)/255.0
val_Y = np.array(val_Y)

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, val_X.shape, val_Y.shape)    #test - delete

train_Y = train_Y.reshape((len(train_Y), 1))
test_Y = test_Y.reshape((len(test_Y), 1))
val_Y = val_Y.reshape((len(val_Y),1))

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, val_X.shape, val_Y.shape)    #test - delete


# In[13]:


print(test_X[0])


# In[14]:


#set data up for CNN

CNN_train_X = train_X.reshape(-1, img_size, img_size, 1)
CNN_test_X = test_X.reshape(-1, img_size, img_size, 1)
CNN_val_X = val_X.reshape(-1, img_size, img_size, 1)

print(CNN_train_X.shape)
print(train_Y.shape)


# In[15]:


#Data Preprocessing & Augmentation 

train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    rotation_range = 15,
    width_shift_range = 0.2, 
    height_shift_range = 0.2,
    shear_range = 0.1, 
    zoom_range = 0.2, 
    horizontal_flip = True)

train_datagen.fit(CNN_train_X)


# In[16]:


#Building CNN Model (random choice of hyperparameters -- just to go over the whole process before optimizing model)
#resulted in surprisingly good accuracy 

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir = log_dir, histogram_freq = 1)

model = keras.Sequential()

model.add(Conv2D(32, (3,3), input_shape = CNN_train_X.shape[1:], activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(.3))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()


# In[17]:


#Compile & Train Model 

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['accuracy'])
            
#epochs = 20 worked pretty well too. 
#epochs = 30 worked very well (accuracy = 87)

model.fit(train_datagen.flow(CNN_train_X, train_Y, batch_size = 32), epochs = 30, validation_data = (CNN_val_X, val_Y), callbacks = [tensorboard])


# In[18]:


#Visualize accuracy & loss w/ tensorboard 
#Evaluate & Analyze

get_ipython().run_line_magic('tensorboard', '--logdir logs/fit/20200826-055504')


# In[19]:


#Check Loss & Accuracy on Test Set 

p = model.predict(CNN_test_X)
threshold = 0.5
p = p > threshold     #convert float values to binary classification values (MAJOR BUG TOOK FOREVER) : https://stackoverflow.com/questions/56073949/should-i-convert-classification-output-to-integer-and-how 

test_loss, test_acc = model.evaluate(CNN_test_X, test_Y)
print("The testing loss and accuracy are :  " + str(round(test_loss, 3)) + " and " + str(round(test_acc*100,2)) + "%")


# In[20]:


#confusion matrix 
con_mat = confusion_matrix(test_Y, p)
print(con_mat)

sns.heatmap(con_mat, annot=True)


# In[21]:


sns.heatmap(con_mat/np.sum(con_mat), annot=True, 
            fmt='.2%', cmap='Blues')


# In[22]:


#Classification Report
print('The Classification Report : \n{}'.format(classification_report(test_Y,p)))


# In[18]:


#Convolutional Neural Network (optimizing model)
#run model w/ various # of dense layers, layer sizes, and # of conv layers 
#and visualize each model's accuracy & loss w/ tensorboard
#find best model. 

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            log_dir = "logs/lung_CNN2/{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard = TensorBoard(log_dir = log_dir, histogram_freq = 1)

            model = keras.Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = CNN_train_X.shape[1:], activation = 'relu'))
            model.add(MaxPool2D(pool_size = (2,2)))

            for c in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3), activation = 'relu'))
                model.add(MaxPool2D(pool_size = (2,2)))

            model.add(Flatten())
            model.add(Dropout(.3))

            for d in range(dense_layer):
                model.add(Dense(layer_size, activation = 'relu'))

            model.add(Dense(1, activation = 'sigmoid'))

            model.summary()

            model.compile(optimizer='adam', 
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

            model.fit(train_datagen.flow(CNN_train_X, train_Y, batch_size = 32), epochs = 20, validation_data = (CNN_val_X, val_Y), callbacks = [tensorboard])
            
            test_loss, test_acc = model.evaluate(CNN_test_X, test_Y)
            print("The testing loss and accuracy are :  " + str(round(test_loss, 3)) + " and " + str(round(test_acc*100,2)) + "%")


# In[ ]:


'''
RESULT: 

with 20 epochs: 
1st place:      2 conv layer, 32 layer size, 2 additional dense layer - test accuracy: 89.58% 
2nd place:      3 conv layer, 64 layer size, 1 additional dense layer - test accuracy: 89.26% 
3rd place:      2 conv layer, 128 layer size, 2 additional dense layer - test accuracy: 88.3% 

--> max test_accuracy is 89.58%

try with fewer epochs: 
2 conv layer, 128 layer size, no additional dense layer - val accuracy: 94.34% @epoch 16 & epoch 19
3 conv layer, 128 layer size, no additional dense layer - val accuracy: 94.34% @epoch 9 & epoch 11 & epoch 13
3 conv layer, 32 layer size, 1 additional dense layer - val accuracy: 94.34% @epoch 9 & epoch 18
3 conv layer, 64 layer size, 1 additional dense layer - val accuracy: 94.34% @epoch 17 & epoch 18
2 conv layer, 128 layer size, 1 additional dense layer - val accuracy: 94.34% @epoch 14
3 conv layer, 32 layer size, 2 additional dense layer - val accuracy: 94.34% @epoch 7
2 conv layer, 64 layer size, 2 additional dense layer - val accuracy: 94.34% @epoch 12 & epoch 13
2 conv layer, 128 layer size, 2 additional dense layer - val accuracy: 94.34% @epoch 16
3 conv layer, 128 layer size, 2 additional dense layer - val accuracy: 94.34% @epoch 17

--> max val_accuracy is 94.34%

concerns: observe exactly repeating val_accuracy numbers. ex) 0.9245, 0.9434 --> maybe because of the small val_test size? 
'''


# In[19]:


get_ipython().run_line_magic('tensorboard', '--logdir logs/lung_CNN2/')

