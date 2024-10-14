# from skorch.dataset import Dataset
import torch
from braindecode.torch_ext.util import np_to_var, var_to_np
from braindecode.datautil.iterators import get_balanced_batches
from numpy.random import RandomState
import torch.nn.functional as F
from torch import optim
import numpy as np
from braindecode.torch_ext.util import set_random_seeds
from braindecode.datautil.signal_target import SignalAndTarget
# from braindecode.models import ShallowFBCSPNet
from CLSTM import ShallowFBCSPLSTM
# from skorch.callbacks import LRScheduler
# from skorch.helper import predefined_split
# from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from utils1 import *
import time
import argparse


parser = argparse.ArgumentParser(description="Please input the number of epoches and the value of early stopping patience.")
parser.add_argument("--numepochs", type=int)
parser.add_argument("--Patience4ES", type=int)
args = parser.parse_args()
Epochs = args.numepochs
Patience = args.Patience4ES


# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0


# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

n_epochs = Epochs
n_classes = 4
lstm_size = 30
lstm_layers = 1
rng = RandomState((2022,3,8))
patience = Patience
epoch_no_improve = 0
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subjects = [1, 2]
save_path = '/home/henrywang/testEEGModels/BCIIV2a/data/'
results = np.zeros((6, 4))
random_seeds = [72, 66, 18, 46, 11]


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
    print(data_teX_c.shape)
    print(data_teY_c.shape)


    train_set = SignalAndTarget(data_trX_c, y=data_trY_c)
    test_set = SignalAndTarget(data_teX_c, y=data_teY_c)
    valid_set = SignalAndTarget(data_vaX_c, y=data_vaY_c)
    min_val_loss = np.Inf

    
    cuda = torch.cuda.is_available()
    set_random_seeds(seed=20200220, cuda = cuda)
    in_chans = train_set.X.shape[1]

    model = ShallowFBCSPLSTM(in_chans=in_chans, n_classes=n_classes, 
                                input_time_length=train_set.X.shape[2], lstm_size=lstm_size, 
                                lstm_layers=lstm_layers,n_filters_time=lstm_size, 
                                n_filters_spat=lstm_size, final_conv_length=4, 
                                pool_time_length=20, pool_time_stride=5).create_network()
    if cuda:
        model.cuda()
    weights_path = "Checkpoints/CLSTM-{0}.hdf5".format(i+1)
    optimizer = optim.Adam(model.parameters())
    time_s = time.time()
    for i_epoch in range(n_epochs):
        i_trials_in_batch = get_balanced_batches(len(train_set.X), rng, shuffle=True,
                                                    batch_size=32)
        # Set model to training mode
        model.train()
        for i_trials in i_trials_in_batch:
            net_in = np_to_var(train_set.X[i_trials][:,:,:,None])
            if cuda:
                # move tensors from cpu to gpu
                net_in = net_in.cuda()
            net_target = np_to_var(train_set.y[i_trials])
            if cuda:
                net_target = net_target.cuda()
                # Remove gradients of last backward pass from all parameters
                optimizer.zero_grad()
                # the type of model parameters should be consistant with that of input (i.e., DoubleTensors)

                outputs = model(net_in.float())
                outputs = torch.mean(outputs, dim=2, keepdim=False)
                # Compute the loss
                loss = F.nll_loss(outputs, net_target)
                # Do the backpropagation
                loss.backward()
                # Update parameters with the optimizer
                optimizer.step()
        # Print some statistics each epoch
        model.eval()
        current_epoch = i_epoch+1
        print("Epoch {:d}".format(current_epoch))
        setname = 'Valid'
        net_in = np_to_var(valid_set.X[:,:,:,None])
        if cuda:
            net_in = net_in.cuda()
        net_target = np_to_var(valid_set.y)
        if cuda:
            net_target = net_target.cuda()
        outputs = model(net_in.float())
        outputs = torch.mean(outputs, dim=2, keepdim=False)
        loss = F.nll_loss(outputs, net_target)
        print("{:6s} Loss: {:.5f}".format(
            setname, float(var_to_np(loss))))
        predicted_labels = np.argmax(var_to_np(outputs), axis=1)
        accuracy = np.mean(valid_set.y  == predicted_labels)
        print("{:6s} Accuracy: {:.1f}%".format(
            setname, accuracy * 100))    

        if loss < min_val_loss:
            epochs_no_improve = 0
            print("val_loss improved from {0} to {1}".format(min_val_loss, loss))
            min_val_loss = loss     
        else:
            epochs_no_improve += 1
            torch.save(model, weights_path)
        if patience == epochs_no_improve:
            print("early stopping!!!")
            break
        else:
            continue
        break        
    time_e = time.time()
    print("Training time:" + str(time_e-time_s))
    model = torch.load(weights_path)
    model.eval()
    net_in = np_to_var(test_set.X[:,:,:,None])
    if cuda:
        net_in = net_in.cuda()
    if cuda: 
        net_target = net_target.cuda()
    
    outputs = model(net_in.float())
    outputs = torch.mean(outputs, dim=2, keepdim=False)
    predicted_labels = np.argmax(var_to_np(outputs), axis=1)
    y_pred_prob = torch.exp(outputs) 
    row_sums = torch.sum(y_pred_prob, 1) # normalization
    row_sums = row_sums.reshape((len(row_sums), 1))
    row_sums = row_sums.repeat(1, n_classes)
    y_test = torch.div( y_pred_prob , row_sums )
    y_pred_prob = var_to_np(y_test)
    acc = 100 * np.mean(test_set.y  == predicted_labels)
    f1 = f1_score(test_set.y, predicted_labels, average='macro')
    auc = roc_auc_score(test_set.y, y_pred_prob, multi_class ='ovr', average='macro')
    kappa = cohen_kappa_score(test_set.y, predicted_labels) 
    results[i, 0] = round(acc, 2)
    results[i, 1] = round(f1, 3)
    results[i, 2] = round(auc, 3)
    results[i, 3] = round(kappa, 3)
results[5, :] = np.round(np.mean(results[:-1], axis = 0), decimals = 3)
np.save('Checkpoints/results_CLSTM.npy', results)
