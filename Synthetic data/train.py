import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from tensorflow.keras.models import load_model
from rich.progress import track
from utils import *
from EEGNet import *

subject_ids = list(range(1, 1744))


# groups = [1, 2, 3, 4, 5, 6]
groups = [1]
epochs = 300
chans = 19
batch_size = 64
folds = range(5)
results = np.zeros((len(groups), len(folds) + 1, 4))
nb_classes = 2
kernels = 1

for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)   
        X_train, y_train = generate_data(train_ids, iscrop = True, istraining=True)
        X_test, y_test = generate_data(test_ids, iscrop = True, istraining = False)
        X_val, y_val = generate_data(val_ids, iscrop = True, istraining = False)
        y_train = onehot_labels(y_train.reshape(-1, 1))
        y_test = onehot_labels(y_test.reshape(-1, 1))
        y_val = onehot_labels(y_val.reshape(-1, 1))
        X_train = np.swapaxes(X_train, 1, 2)
        X_test = np.swapaxes(X_test, 1, 2)
        X_val = np.swapaxes(X_val, 1, 2)
        samples = X_train.shape[2]
        X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)
        X_val = X_val.reshape(X_val.shape[0], chans, samples, kernels)
        model = EEGNet(nb_classes = 2, Chans = 19, 
                       dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                       dropoutType = 'Dropout')
        weights_path = "Checkpoints/EEGNet-group{0}-{1}.hdf5".format(group_id, i)
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])
        earlystopping = EarlyStopping(monitor='val_loss', patience=30)
        model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor = 'val_loss', 
                                           verbose = 1, save_best_only = True)
        callback_list = [model_checkpoint, earlystopping]
        fittedModel = model.fit(X_train, y_train, epochs = epochs, 
                                batch_size = batch_size, validation_data = 
                                (X_val, y_val), verbose= 2, shuffle = True, 
                                callbacks=callback_list)
        model = load_model('Checkpoints/EEGNet-group{0}-{1}.hdf5'.format(group_id, i))
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis = 1)
        y_true = np.argmax(y_test, axis = 1)
        acc = 100 * accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        kappa = cohen_kappa_score(y_true, y_pred)
        results[group_id-1, i, 0] = round(acc, 2)
        results[group_id-1, i, 1] = round(f1, 3)
        results[group_id-1, i, 2] = round(auc, 3)
        results[group_id-1, i, 3] = round(kappa, 3)
    results[group_id-1, len(folds), :] = np.round(np.mean(results[group_id-1][:-1], axis = 0), decimals = 3)
np.save('Checkpoints/results.npy', results)
fold_num = results.shape[1]
CGREEN = '\033[92m'
CEND = '\033[0m'
print('\nFinal results:')
for i in range(fold_num):
    if i == fold_num - 1:
        print('Average classification accuracy is ' + CGREEN + str(results[0][i][0]) + '%' + CEND)
    else:
        print('Classification accuracy for the evaluation set-{0} is '.format(i+1) + CGREEN + str(results[0][i][0])+'%' + CEND)
