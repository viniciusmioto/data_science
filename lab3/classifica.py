#!/usr/bin/env python3

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

X, y = load_svmlight_file('train.txt')
X_test, y_test = load_svmlight_file('test.txt')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

## save for the confusion matrix
label = y_test

## converts the labels to a categorical one-hot-vector
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=2)
y_val = keras.utils.np_utils.to_categorical(y_val, num_classes=2)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes=2)


model = Sequential()
# Dense(50) is a fully-connected layer with 50 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 300-dimensional vectors.
model.add(Dense(50, activation='relu', input_dim=300))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128, verbose=1)

loss, acc = model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
classes=np.argmax(y_pred,axis=1)

cm = confusion_matrix(label, classes)
print (cm)

print ('Loss    : ', loss)
print ('Accuracy: ', acc)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






