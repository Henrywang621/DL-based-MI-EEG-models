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
results = np.zeros((len(groups), len(folds) + 1, 4))

for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)   
#         np.save('Checkpoints/test_ids{0}-{1}.npy'.format(group_id, i), test_ids)
        X_train, y_train = load_data(train_ids, group_id, istraining = True)
        X_test, y_test = load_data(test_ids, group_id, istraining = False)
        X_val, y_val = load_data(val_ids, group_id, istraining = False)
        X_train, y_train = prepare_data(np.array(X_train), np.array(y_train).reshape(-1, 1))
        X_test, y_test = prepare_data(np.array(X_test), np.array(y_test).reshape(-1, 1))
        X_val, y_val = prepare_data(np.array(X_val), np.array(y_val).reshape(-1, 1))
        X_train = np.expand_dims(X_train, axis = -1)
        X_test = np.expand_dims(X_test, axis = -1)
        X_val = np.expand_dims(X_val, axis = -1)
        model = EEGNet(nb_classes = 2, Chans = 64, 
                       dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                       dropoutType = 'Dropout')
             
        weights_path = "Checkpoints/EEGNet-group{0}-{1}.hdf5".format(group_id, i)
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])
        earlystopping = EarlyStopping(monitor='val_loss', patience=30)
        model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor='val_loss', 
                                           verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=log_path)
        callback_list = [model_checkpoint, earlystopping, tensorboard]
        fittedModel = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, y_val), 
                                verbose= 2, shuffle = False, callbacks=callback_list)
        model = load_model('Checkpoints/EEGNet-group{0}-{1}.hdf5'.format(group_id, i))
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis = 1)
        y_true = np.argmax(y_test, axis = 1)
        acc = 100 * accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)
        kappa = cohen_kappa_score(y_true, y_pred)
        results[group_id-1, i, 0] = round(acc, 2)
        results[group_id-1, i, 1] = round(f1, 3)
        results[group_id-1, i, 2] = round(auc, 3)
        results[group_id-1, i, 3] = round(kappa, 3)
    results[group_id-1, len(folds), :] = np.round(np.mean(results[group_id-1][:-1], axis = 0), decimals = 3)
np.save('Checkpoints/results.npy', results)
