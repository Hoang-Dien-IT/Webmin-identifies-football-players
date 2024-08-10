import numpy as np
import os
import cv2
import tensorflow as tf
import time

from keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


folder = '/Train_data/'

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

data = np.array(data)
labels = np.array(labels)
data = data.reshape((-1, 100, 100, 1))
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
X_train = X_train / 255.0
X_val = X_val / 255.0

lb = LabelEncoder()
Y_train = to_categorical(lb.fit_transform(Y_train), num_classes=len(categories))
Y_val = to_categorical(lb.fit_transform(Y_val), num_classes=len(categories))


# Hàm tạo mô hình với dropout
def create_model_with_dropout(neurons=512, dropout_rate=0.2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100, 100, 1)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(neurons))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(Dropout(dropout_rate))  # Thêm dropout
    model.add(Dense(len(categories)))
    model.add(tf.keras.layers.Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


# Tìm kiếm Grid Search thủ công
param_grid = {'neurons': [256, 512, 1024],
              'dropout_rate': [0.1, 0.2, 0.3]}

best_score = 0
best_params = {}

for neurons in param_grid['neurons']:
    for dropout_rate in param_grid['dropout_rate']:
        model = create_model_with_dropout(neurons=neurons, dropout_rate=dropout_rate)
        model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)
        val_preds = model.predict(X_val)
        val_preds = np.argmax(val_preds, axis=1)
        Y_val_labels = np.argmax(Y_val, axis=1)
        accuracy = accuracy_score(Y_val_labels, val_preds)

        print(f"Accuracy for neurons={neurons}, dropout_rate={dropout_rate}: {accuracy}")

        if accuracy > best_score:
            best_score = accuracy
            best_params = {'neurons': neurons, 'dropout_rate': dropout_rate}

print(f"Best: {best_score} using {best_params}")

end_time = time.time()
training_time = end_time - start_time
print("Training time: {:.2f} seconds".format(training_time))

