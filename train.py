import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
import h5py
K.set_image_dim_ordering('th')

np.random.seed(1337)  # for reproducibility
batch_size = 128
nb_classes = 10
epochs = 20


def convmodel():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), activation='relu', padding='valid', input_shape=(1, 55, 30), name='conv1'))
    model.add(MaxPooling2D((2, 2), name='pool1'))

    model.add(Conv2D(15, (3, 3), activation='relu',  name='conv2'))
    model.add(MaxPooling2D((2, 2), name='pool2'))

    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(128, activation='relu', name='fc1'))
    model.add(Dense(50, activation='relu', name='fc2'))
    model.add(Dense(10, activation='softmax', name='predictions'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# the data, shuffled and split between train and test sets
f_traindata = h5py.File('./Network/data/train_data.h5', 'r')
f_trainlabel = h5py.File('./Network/data/train_label.h5', 'r')
f_valdata = h5py.File('./Network/data/val_data.h5', 'r')
f_vallabel = h5py.File('./Network/data/val_label.h5', 'r')


X_train = f_traindata['data'][:]
y_train = f_trainlabel['label'][:]

X_test = f_valdata['data'][:]
y_test = f_vallabel['label'][:]


X_train = X_train.reshape(X_train.shape[0], 1,55,30)
X_test = X_test.reshape(X_test.shape[0], 1,55,30)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = convmodel()
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model_json = model.to_json()
with open("./Network/model/digit_CNN_2.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("./Network/model/digit_CNN_2.h5")
print("Saved model to disk")