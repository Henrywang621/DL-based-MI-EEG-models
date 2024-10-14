import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
import numpy as np
from tensorflow.keras import optimizers
from utils import *
from MIN2Net import *
import mne
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import argparse
from loss import *



'''
 To obtain more robust results, I evaluate the model on 5 different sets, each of which 
 contain different training sets, validation sets and test sets.


 '''

 # Ref: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)]) # Notice here
        # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

parser = argparse.ArgumentParser(description="Please input the number of epoches and the value of early stopping patience.")
parser.add_argument("--numepochs", type=int)
parser.add_argument("--Patience4ES", type=int)
args = parser.parse_args()
Epochs = args.numepochs
Patience = args.Patience4ES

save_path = '/home/henrywang/testEEGModels/BCIIV2a/data/'
random_seeds = [72, 66, 18, 46, 11]
# results = np.zeros((len(subjects), len(folds) + 1, 4))
nclasses = 4
chans = 22
epochs = Epochs
batch_size = 64
results = np.zeros((6, 4))


for i in range(5):
    subjs_tr, subjs_va, subjs_te = split_datasets(seed=random_seeds[i])
    # data_X_tr = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 1.171875, npad='auto') for i in subjs_tr]
    data_X_tr = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 2.8125, npad='auto') for i in subjs_tr]
    data_trX_c = np.vstack(data_X_tr)

    # Clear the GPU memory of the variable data_X_tr
    del data_X_tr

    # data_X_va = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(j)), down = 1.171875, npad='auto') for j in subjs_va]
    data_X_va = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(j)), down = 2.8125, npad='auto') for j in subjs_va]
    data_vaX_c = np.vstack(data_X_va)
    
    # Clear the GPU memory of the variable data_X_tr
    del data_X_va   

    # data_X_te = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 1.171825, npad='auto') for i in subjs_te]
    data_X_te = [mne.filter.resample(np.load(save_path+'S{0}_X.npy'.format(i)), down = 2.8125, npad='auto') for i in subjs_te]
    data_teX_c = np.vstack(data_X_te)

    # Clear the GPU memory of the variable data_X_tr
    del data_X_te

    data_y_tr = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_tr]
    data_trY_c = np.vstack(data_y_tr)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_tr

    data_y_te = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_te]
    data_teY_c = np.vstack(data_y_te)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_te

    data_y_va = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_va]
    data_vaY_c = np.vstack(data_y_va)

    # Clear the GPU memory of the variable data_X_tr
    del data_y_va

    data_trX_c = std(data_trX_c)
    data_trX_c = np.swapaxes(data_trX_c, 1, 2)
    data_trX_c = np.expand_dims(data_trX_c, axis = 1)
    data_vaX_c = std(data_vaX_c)
    data_vaX_c = np.swapaxes(data_vaX_c, 1, 2)
    data_vaX_c = np.expand_dims(data_vaX_c, axis = 1)
    data_X_te = std(data_teX_c)
    data_teX_c = np.swapaxes(data_teX_c, 1, 2)
    data_teX_c = np.expand_dims(data_teX_c, axis = 1)

    # print(data_trX_c.shape)
    # print(data_trY_c.shape)
    # print(data_vaX_c.shape)
    # print(data_vaY_c.shape)    
    # print(data_teX_c.shape)
    # print(data_teY_c.shape)    



    # weights_path = "Checkpoints/Min2Net-{0}.hdf5".format(i+1)
    weights_path = "Checkpoints/Min2Net-{0}".format(i+1)
    model = MIN2Net(epochs= Epochs, patience=Patience, weights_path=weights_path)
    
    # model.compile(loss='categorical_crossentropy', optimizer='adam', 
    #                 metrics = ['accuracy'])
    # earlystopping = EarlyStopping(monitor='val_loss', patience=Patience)        
    # model_checkpoint = ModelCheckpoint(filepath = weights_path, monitor = 'val_loss', 
    #                                     verbose = 1, save_best_only = True)
    # callback_list = [model_checkpoint, earlystopping]
    fittedModel = model.fit(data_trX_c, data_trY_c, data_vaX_c, data_vaY_c)
    # fittedModel = model.fit(data_trX_c, [data_trX_c, data_trY_c, data_trY_c], 
    #                         epochs = epochs, batch_size = batch_size, 
    #                         validation_data = (data_vaX_c, [data_vaX_c, data_vaY_c, data_vaY_c]), 
    #                         verbose= 2, shuffle = True, callbacks=callback_list)

    model = load_model('Checkpoints/Min2Net-{0}'.format(i+1), custom_objects={'inner_triplet_loss_objective': triplet_loss})
    # Y, evaluation = model.predict(data_teX_c, data_teY_c)
    _, _, y_pred_prob = model.predict(data_teX_c)

    y_pred = np.argmax(y_pred_prob, axis = 1)
    y_true = data_teY_c

    # print(f"y_true shape: {y_true.shape}")
    # print(f"y_score shape: {y_score.shape}")
    acc = 100 * np.mean(y_true  == y_pred)
    f1 = f1_score(y_true, y_pred, average='macro') 
    auc = roc_auc_score(y_true, y_pred_prob, multi_class ='ovr', average='macro')
    kappa = cohen_kappa_score(y_true, y_pred) 
    results[i, 0] = round(acc, 2)
    results[i, 1] = round(f1, 3)
    results[i, 2] = round(auc, 3)
    results[i, 3] = round(kappa, 3)
results[5, :] = np.round(np.mean(results[:-1], axis = 0), decimals = 3)
np.save('Checkpoints/results_Min2Net.npy', results)
