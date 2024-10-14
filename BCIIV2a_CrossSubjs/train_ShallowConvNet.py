from skorch.dataset import Dataset
import torch
import numpy as np
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from utils1 import *
import time
import argparse

# from braindecode.datasets import MOABBDataset
# from braindecode.preprocessing import create_windows_from_events
# from braindecode.preprocessing import (
#     exponential_moving_standardize, preprocess, Preprocessor, scale)

# These values we found good for shallow network:

parser = argparse.ArgumentParser(description="Please input the number of epoches and the value of early stopping patience.")
parser.add_argument("--numepochs", type=int)
parser.add_argument("--Patience4ES", type=int)
args = parser.parse_args()
Epochs = args.numepochs
Patience = args.Patience4ES

lr = 0.0625 * 0.01
weight_decay = 0
batch_size = 64
n_epochs = Epochs

save_path = '/home/henrywang/testEEGModels/BCIIV2a/data/'
random_seeds = [72, 66, 18, 46, 11]
results = np.zeros((6, 4))


for i in range(5):
    subjs_tr, subjs_va, subjs_te = split_datasets(seed=random_seeds[i])
    data_X_tr = [np.load(save_path+'S{0}_X.npy'.format(i)) for i in subjs_tr]
    data_trX_c = np.vstack(data_X_tr)

    # Clear the GPU memory of the variable data_X_tr
    del data_X_tr

    data_X_va = [np.load(save_path+'S{0}_X.npy'.format(j)) for j in subjs_va]
    data_vaX_c = np.vstack(data_X_va)
    
    # Clear the GPU memory of the variable data_X_tr
    del data_X_va   

    data_X_te = [np.load(save_path+'S{0}_X.npy'.format(j)) for j in subjs_te]
    data_teX_c = np.vstack(data_X_te)
    
    # Clear the GPU memory of the variable data_X_tr
    del data_X_te   

    data_y_tr = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_tr]
    data_trY_c = np.squeeze(np.vstack(data_y_tr))

    # Clear the GPU memory of the variable data_X_tr
    del data_y_tr

    data_y_va = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_va]
    data_vaY_c = np.squeeze(np.vstack(data_y_va))

    # Clear the GPU memory of the variable data_X_tr
    del data_y_va

    data_y_te = [np.load(save_path+'S{0}_y.npy'.format(i)).reshape(-1, 1) for i in subjs_te]
    data_teY_c = np.squeeze(np.vstack(data_y_te))

    # Clear the GPU memory of the variable data_X_tr
    del data_y_te

    print(data_trX_c.shape)
    print(data_trY_c.shape)

    train_set = Dataset(data_trX_c, data_trY_c)
    valid_set = Dataset(data_vaX_c, data_vaY_c)
    test_set = Dataset(data_teX_c, data_teY_c)
    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    # Send model to GPU
    if cuda:
        model.cuda()
    
    cp = Checkpoint(dirname='Checkpoints', fn_prefix = 'SC-{0}'.format(i))
    early_stopping = EarlyStopping(patience=Patience)
    train_acc = EpochScoring(scoring='accuracy', on_train=True, 
                                name='train_acc', lower_is_better=False)
    valid_acc = EpochScoring(scoring='accuracy', on_train=False, 
                                name='valid_acc', lower_is_better=False)
    callbacks = [
        ('cp', cp),
        ('patience', early_stopping),
        ('train_acc', train_acc),
        ('valid_acc', valid_acc),
    ]

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size = batch_size,
        callbacks = callbacks,
        device = device,
    )
    # Model training for a specified number of epochs. `y` is None as it is already supplied
    # in the dataset.

    time_s = time.time()
    clf.fit(X= train_set, y=None, epochs=n_epochs)
    time_e = time.time()
    print("Training time:" + str(time_e-time_s))

            
    clf.initialize()
    clf.load_params(checkpoint=cp)
    time_s = time.time()
    predicted_labels = clf.predict(test_set)
    time_e = time.time()
    print("inference time:" + str(time_e-time_s))
        
    y_pred_prob = clf.predict_proba(test_set)
    y_pred_prob = torch.exp(torch.tensor(y_pred_prob)) 
    row_sums = torch.sum(y_pred_prob, 1) # normalization
    row_sums = row_sums.reshape((len(row_sums), 1))
    row_sums = row_sums.repeat(1, n_classes)
    y_pred_prob = torch.div( y_pred_prob , row_sums )
    y_pred_prob = y_pred_prob.cpu().detach().numpy()
    y_true = data_teY_c
    acc = 100 * np.mean(test_set.y  == predicted_labels)
    f1 = f1_score(test_set.y, predicted_labels, average='macro')
    auc = roc_auc_score(test_set.y, y_pred_prob, multi_class ='ovr', average='macro')
    kappa = cohen_kappa_score(test_set.y, predicted_labels) 
    results[i, 0] = round(acc, 2)
    results[i, 1] = round(f1, 3)
    results[i, 2] = round(auc, 3)
    results[i, 3] = round(kappa, 3)
results[5, :] = np.round(np.mean(results[:-1], axis = 0), decimals = 3)
np.save('Checkpoints/results_SC.npy', results)
print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")  