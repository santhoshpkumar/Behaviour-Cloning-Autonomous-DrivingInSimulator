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


## Helper Functions

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

## End Helper Functions Block


-

model = Sequential()

'''
# MODEL 7
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('tanh'))
model.add(Dense(64))
model.add(Dense(1))
'''

# MODEL 8
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (160, 320, 3)))
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

# Compile and train the model, 

#model.compile(optimizer=Adam(lr=1e-4), loss='mse')
model.compile(loss='mse', optimizer='adam')

checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

model.fit(X_train,y_train, validation_split=0.2, nb_epoch=5, shuffle=True, verbose=1, callbacks=[checkpoint])

print("Complete Model Building")
model.save('model.h5')