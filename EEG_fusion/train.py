import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from utils1 import *
from EEGModels import EEGNet_fusion

subject_ids = list(range(1, 110))
# Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
remove_ids = [88, 89, 92, 100, 104]

# remove subjects'data sampled at 128Hz
for id in remove_ids:
    subject_ids.remove(id)


# groups = [1, 2, 3, 4, 5, 6]
groups = [1]
epochs = 300
chans = 64
kernels = 1
batch_size = 64
folds = range(5)
log_path = 'log/'
nb_classes=2

for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)   
#         np.save('Checkpoints/test_ids{0}-{1}.npy'.format(group_id, i), test_ids)
        X_train, y_train = load_data(train_ids, group_id)
        X_val, y_val = load_data(val_ids, group_id)
        X_train, y_train = prepare_data(np.array(X_train), np.array(y_train).reshape(-1, 1))
        X_val, y_val = prepare_data(np.array(X_val), np.array(y_val).reshape(-1, 1))
        samples = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], samples, chans, kernels)
        X_val = X_val.reshape(X_val.shape[0], samples, chans, kernels)
        model = EEGNet_fusion(nb_classes, chans, Samples= 960, dropoutRate=0.5, 
                              norm_rate=0.25, dropoutType='Dropout', cpu=True)  
        weights_path = "Checkpoints/EEGNet_fusion-group{0}-{1}.hdf5".format(group_id, i)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), 
                      metrics = ['accuracy'])
        earlystopping = EarlyStopping(monitor='val_loss', patience=30)
        model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_loss', 
                                           verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=log_path)
        callback_list = [model_checkpoint, earlystopping, tensorboard]
        fittedModel = model.fit([X_train, X_train, X_train], y_train, epochs = epochs, 
                                batch_size = batch_size, validation_data = 
                                ([X_val, X_val, X_val], y_val), verbose= 2, shuffle = True, 
                                callbacks=callback_list)