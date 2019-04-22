from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

def get_model(features_shape):
    # (number of freqs x number of frames in a segment x number of channels)
    input_shape = (features_shape[1],features_shape[2], 1)

    model = Sequential()

    model.add(Conv2D(32, (5, 5),
            input_shape=input_shape,
            activation = "relu",
            padding = "same"))
    # model.add(MaxPooling2D(pool_size=(4, 4)))

    model.add(Conv2D(64, (5, 5),
            activation = "relu",
            padding = "same"))

    model.add(Conv2D(1, (10, 10),
            activation = "relu",
            padding = "same"))

    adam = Adam(lr=0.0003)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['mean_absolute_error'])
    model.summary()
    return model