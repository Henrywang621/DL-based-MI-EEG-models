import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from utils1 import *
from ETENet import create_raw_model

subject_ids = list(range(1, 110))
# Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
remove_ids = [88, 89, 92, 100, 104]

# remove subjects'data sampled at 128Hz
for id in remove_ids:
    subject_ids.remove(id)


# groups = [1, 2, 3, 4, 5, 6]
# groups = list(range(1, 11))
groups = [1]
epochs = 200
chans = 64
kernels = 1
batch_size = 16
folds = range(5)
log_path = 'log1/'
nclasses=2

for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)   
        X_train, y_train = load_data(train_ids, group_id)
        X_val, y_val = load_data(val_ids, group_id)
        X_train, y_train = prepare_data(np.array(X_train), np.array(y_train).reshape(-1, 1))
        X_val, y_val = prepare_data(np.array(X_val), np.array(y_val).reshape(-1, 1))
        samples = X_train.shape[1]
        X_train = X_train.reshape(X_train.shape[0], samples, chans, kernels)
        X_val = X_val.reshape(X_val.shape[0], samples, chans, kernels)
        model = create_raw_model(chans, nclasses, trial_length=960, l1=0) 
        weights_path = "Checkpoints/ETENet-group{0}-{1}.hdf5".format(group_id, i)
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])
        earlystopping = EarlyStopping(monitor='val_loss', patience=30)
        model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_loss', 
                                           verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=log_path)
        callback_list = [model_checkpoint, earlystopping, tensorboard]
        fittedModel = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,                                               validation_data = (X_val, y_val), verbose= 2, shuffle = True,                                             callbacks=callback_list)