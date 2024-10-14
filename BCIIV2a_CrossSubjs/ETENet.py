from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Dropout 
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint

def create_raw_model(nchan, nclasses, trial_length=720, l1=0):
    """
    CNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(Conv2D(40, (1, nchan), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="valid"))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model