import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D

import kapre
from kapre.utils import Normalization2D
from kapre.time_frequency import Spectrogram



class CNN_STFT(tf.keras.Model):

    def create_model(self, input_shape, dropout=0.5, print_summary=False):

        # basis of the CNN_STFT is a Sequential network
        model = Sequential()

        # spectrogram creation using STFT
        model.add(Spectrogram(n_dft = 128, n_hop = 16, input_shape = input_shape,
                  return_decibel_spectrogram = False, power_spectrogram = 2.0,
                  trainable_kernel = False, name = 'static_stft'))
        model.add(Normalization2D(str_axis = 'freq'))

        # Conv Block 1
        model.add(Conv2D(filters = 24, kernel_size = (12, 12),
                         strides = (1, 1), name = 'conv1',
                         padding = 'same'))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid',
                               data_format = 'channels_last'))

        # Conv Block 2
        model.add(Conv2D(filters = 48, kernel_size = (8, 8),
                         name = 'conv2', padding = 'same'))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid',
                               data_format = 'channels_last'))

        # Conv Block 3
        model.add(Conv2D(filters = 96, kernel_size = (4, 4),
                         name = 'conv3', padding = 'same'))
        model.add(BatchNormalization(axis = 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2),
                               padding = 'valid',
                               data_format = 'channels_last'))
        model.add(Dropout(dropout))

        # classificator
        model.add(Flatten())
        model.add(Dense(4))  # two classes only
        model.add(Activation('softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])

        # assign model and return
        self.model = model
        return model