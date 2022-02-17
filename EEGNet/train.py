import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from utils import *
from EEGNet8_2 import EEGNet

subject_ids = list(range(1, 110))
# Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
remove_ids = [88, 89, 92, 100, 104]

# remove subjects'data sampled at 128Hz
for id in remove_ids:
    subject_ids.remove(id)


# groups = [1, 2, 3, 4, 5, 6]
groups = [1]
epochs = 200
chans = 64
kernels = 1
batch_size = 16
folds = range(5)
log_path = 'log/'

for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)   
#         np.save('Checkpoints/test_ids{0}-{1}.npy'.format(group_id, i), test_ids)
        X_train, y_train = load_data(train_ids, group_id, istraining = True)
        X_val, y_val = load_data(val_ids, group_id, istraining = False)
        X_train, y_train = prepare_data(np.array(X_train), np.array(y_train).reshape(-1, 1))
        X_val, y_val = prepare_data(np.array(X_val), np.array(y_val).reshape(-1, 1))
        samples = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_val = X_val.reshape(X_val.shape[0], chans, samples, kernels)
        model = EEGNet(nb_classes = 2, Chans = 64, 
                       dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                       dropoutType = 'Dropout')
             
        weights_path = "Checkpoints/EEGNet-group{0}-{1}.hdf5".format(group_id, i)
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])
        earlystopping = EarlyStopping(monitor='val_loss', patience=20)
        model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_loss', 
                                           verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=log_path)
        callback_list = [model_checkpoint, earlystopping, tensorboard]
        fittedModel = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,                                                 class_weight = cweights, validation_data = (X_val, y_val), 
                                verbose= 2, shuffle = False, callbacks=callback_list)
