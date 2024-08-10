import numpy as np
import os
import cv2
import tensorflow as tf
import time

from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.preprocessing import LabelEncoder
folder = 'Train_data/'

categories = ["{:05d}".format(number) for number in range(60)]

data = []
labels = []
imagePaths = []

for k, category in enumerate(categories):
    for f in os.listdir(folder + category):
        imagePaths.append([folder + category + '/' + f, k])

start_time = time.time()
for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(src=image, dsize=(100, 100))
    image = np.array(image)
    data.append(image)
    label = imagePath[1]
    labels.append(label)

data1 = np.array(data)
labels = np.array(labels)
data1 = data1.reshape((-1, 100, 100, 1))
X_train = data1 / 255.0

lb = LabelEncoder()
Y_train = to_categorical(lb.fit_transform(labels), num_classes=len(categories))

Model = Sequential()
shape = (100, 100, 1)
Model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(Conv2D(32, (3, 3), padding="same"))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Conv2D(64, (3, 3), padding="same"))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(MaxPooling2D(pool_size=(2, 2)))
Model.add(Flatten())
Model.add(Dense(512))
Model.add(tf.keras.layers.Activation("relu"))
Model.add(Dense(len(categories)))
Model.add(tf.keras.layers.Activation("softmax"))

Model.summary()

Model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

print("Start training model")
Model.fit(X_train, Y_train, batch_size=32, epochs=45, verbose=1)

end_time = time.time()
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))

Model.save("soccer_face_model.h5")
print("Saved model successfully: soccer_face_model.h5")
