import os
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from random import shuffle

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                current_path = '../data/IMG/'

                steering_center = float(batch_sample[3])
                correction = 0.3 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_center = cv2.imread(current_path+line[0].split('/')[-1])
                image_left = cv2.imread(current_path+line[1].split('/')[-1])
                image_right = cv2.imread(current_path+line[2].split('/')[-1])

                # add images and angles to data set
                images.extend((image_center, image_left, image_right))
                angles.extend((steering_center, steering_left, steering_right))

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=1000)
validation_generator = generator(validation_samples, batch_size=1000)

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("Loading images data....")
images = []
measurements = []
current_path = './data/IMG/' 
for line in lines:
    steering_center  = float(line[3])
    correction = 0.3 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    image_center = cv2.imread(current_path+line[0].split('/')[-1])
    image_left = cv2.imread(current_path+line[1].split('/')[-1])
    image_right = cv2.imread(current_path+line[2].split('/')[-1])

    # add images and angles to data set
    images.extend((image_center, image_left, image_right))
    measurements.extend((steering_center, steering_left, steering_right))
print("Completed loading images data....")
	  
print("Augmenting images data....")
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
print("Complete augmenting images data....")
print("Model Training and Building")
	  
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = Sequential()
model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
#Normalize the data.
#model.add( Lambda( lambda x: x/127.5 - 1.) )
model.add( Lambda( lambda x: x/255. - 0.5 ) )

# Nvidia Network
# Convolution Layers
model.add( Convolution2D( 24, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 36, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 48, 5, 5, subsample=(2,2), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
model.add( Convolution2D( 64, 3, 3, subsample=(1,1), activation = 'relu' ) )
# Flatten for transition to fully connected layers.
model.add( Flatten() )
# Fully connected layers
model.add( Dense( 100 ) )
model.add(Activation('elu'))
model.add(Dropout(0.5)) # I added this dropout layer to avoid overfitting. 

model.add( Dense( 50 ) )
#model.add(Activation('tanh'))
#model.add(Activation('sigmoid'))
model.add(Activation('elu'))
#model.add(Activation('relu'))
model.add( Dense( 10 ) )

#model.add(Activation('tanh'))
#model.add(Activation('sigmoid'))
model.add(Activation('elu'))
#model.add(Activation('relu'))
model.add( Dense( 1 ) )

model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

''' 
Using the generator I am unable to get a model with low validation error and my model architecture.
I have a titan x and 1080TI cards, so now sure why I have to worry about memory when I have the capability to get 
the model built without having to use a generator. I have included the generator code to showcase that I have tried it.
But it does not get me the results I want.


model.fit_generator(train_generator, 
					samples_per_epoch= len(train_samples*3), 
					validation_data=validation_generator,
					nb_val_samples=len(validation_samples*3),
					nb_epoch=10,
					callbacks=[checkpoint])
'''


model.fit(X_train,y_train, validation_split=0.2, nb_epoch=5, shuffle=True, verbose=1, callbacks=[checkpoint])

print("Complete Model Building")
model.save('model.h5')