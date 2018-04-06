import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam

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

'''
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
'''

model.add( Cropping2D( cropping=( (50,20), (0,0) ), input_shape=(160,320,3)))
#Normalize the data.
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
model.add(Dropout(0.5)) # I added this dropout layer myself, because the previous 
                        # fully connected layers has a lot of free parameters 
                        # and seems like the layer most in danger of overfitting. 
model.add( Dense( 50 ) )
model.add( Dense( 10 ) )
model.add( Dense( 1 ) )

# Compile and train the model, 

#model.compile(optimizer=Adam(lr=1e-4), loss='mse')
model.compile(loss='mse', optimizer='adam')

model.fit(X_train,y_train, validation_split=0.2, nb_epoch=10, shuffle=True, verbose=1)
print("Complete Model Building")
model.save('model.h5')