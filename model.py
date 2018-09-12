import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

samples = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

current_path = './data/'
correction = 0.2

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                center_name = current_path + batch_sample[0].strip()
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                left_name = current_path + batch_sample[1].strip()
                left_image = cv2.imread(left_name)
                left_angle = center_angle + correction
                right_name = current_path + batch_sample[2].strip()
                right_image = cv2.imread(right_name)
                right_angle = center_angle - correction

                images.extend([center_image, left_image, right_image])
                angles.extend([center_angle, left_angle, right_angle])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


from keras.optimizers import Adam
adam = Adam(lr=0.001, decay=0.75)
model.compile(loss='mse', optimizer=adam)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, \
                    nb_val_samples=len(validation_samples), nb_epoch=20)

model.save('model.h5')
        


