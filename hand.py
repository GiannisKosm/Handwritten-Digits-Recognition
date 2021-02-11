import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.datasets import mnist

#load the data and split them into train/test
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#extract 0 and 1 digits ONLY
train_filter = np.where((Y_train == 0 ) | (Y_train == 1))
test_filter = np.where((Y_test == 0) | (Y_test == 1))

X_train, Y_train = X_train[train_filter], Y_train[train_filter]
X_test, Y_test = X_test[test_filter], Y_test[test_filter]

print("X_train shape", X_train.shape)
print("Y_train shape", Y_train.shape)
print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)

# build the input from the 28x28 pixels
X_train = X_train.reshape(12665, 784) # the first parameter is extracted from line 19
X_test = X_test.reshape(2115, 784) # the first parameter is extracted from line 21
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize the data from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

#number of classes
classes=2

#one-hot encoding 
print("Shape before one-hot encoding: ", Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, classes)
Y_test = np_utils.to_categorical(Y_test, classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# mlp construction
model = Sequential()
model.add(Dense(512, input_shape=(784,))) # input flatten 28x28 pixels
model.add(Activation('relu'))                            
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(2)) # number of output equals to number of classes
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') # adam gradient descent algorithm is used for weights

#initialize parameters
history = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=2, validation_data=(X_test, Y_test)) # the model is fit over 10 epochs with updates every 128 images

# save the model
save = "/Users/Desktop/save"
name = 'hand_digits.h5'
path = os.path.join(save,name)
model.save(path)
print('Saved trained model at %s ' % path)

load = load_model('/Users/Desktop/save/hand_digits.h5')

#model overall evaluation
results = load.evaluate(X_test, Y_test)

print("Loss", results[0])
print("Accuracy", results[1])

# predict first 7 numbers 
predictions = load.predict(X_test[:7])
print(np.argmax(predictions,axis=1))

#evaluate first 7 numbers with matplotlib
for i in range(0,7):
    image = X_test[i]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28,28))
    plt.imshow(pixels,cmap='gray')
    plt.show()


