# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:33:15 2019

@author: mikey
"""

import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from selenium import webdriver
import time
import easygui
import win32ui

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 4, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.load_weights('classifier_weights.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
##########################################################################
file = easygui.fileopenbox()
print (file)

test_image = image.load_img(file,target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

resind0 = str(list(np.reshape(np.asarray(result[0][0]), (1, np.size(result[0][0])))[0]))[1:-1]
resind1 = str(list(np.reshape(np.asarray(result[0][1]), (1, np.size(result[0][1])))[0]))[1:-1]
resind2 = str(list(np.reshape(np.asarray(result[0][2]), (1, np.size(result[0][2])))[0]))[1:-1]
resind3 = str(list(np.reshape(np.asarray(result[0][3]), (1, np.size(result[0][3])))[0]))[1:-1]

if resind0 == '1.0':
    prediction = 'Brick corner 1x2x2'
elif resind1 == '1.0':
    prediction = 'Brick 2x2'
elif resind2 == '1.0':
    prediction = 'Brick 1x2'
else:
    prediction = 'Flat Tile 1x2'

print (prediction)

product = prediction

options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument("--test-type")
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)
driver.get('https://shop.lego.com/en-IE/Pick-a-Brick')


time.sleep(2)

brickNameBox = driver.find_element_by_id("brick_name_Brick Name_pab-label-brick-name")

brickNameBox.send_keys(product, u'\ue007')

time.sleep(2)
price = driver.find_element_by_xpath(".//*[@id='main-content']/div/main/div/div[2]/div/section[1]/div/div[2]/ul/li[1]/div/div/div[2]/div[2]/span").text

driver.close()

win32ui.MessageBox("The brick you selected was: " + product + ". The Price is " + price, "Lego Piece Recognition")