
# coding: utf-8

# In[7]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 4, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = train_datagen.flow_from_directory('C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

from IPython.display import display
from PIL import Image

classifier.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=10,
        validation_data=test_set,
        validation_steps=80)
classifier.save_weights('classifier_weights.h5')


# In[108]:


#'C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid/3004 Brick 1x2/0005.png'
#C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid/3003 Brick 2x2/0001.png
#C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid/2357 Brick Corner 1x2x2/201706171206-0001.png
#C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid/3069 Flat Tile 1x2/0001.png
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from IPython.display import display
from PIL import Image
theImage = 'C:/Users/mikey/OneDrive/Desktop/Lego_Tester_Images/Valid/2357 Brick Corner 1x2x2/201706171206-0001.png'
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
test_image = image.load_img(theImage,
                            target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
classes = training_set.class_indices

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

