from skorch.dataset import Dataset
import torch
from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from utils import *

subject_ids = list(range(1, 110))
# Damaged recordings (#88, #89, #92, #100 and #104) need to be removed.
remove_ids = [88, 89, 92, 100, 104]

groups = [1]

# remove subjects'data sampled at 128Hz
for id in remove_ids:
    subject_ids.remove(id)
    
# These values we found good for shallow network:
lr = 0.0625 * 0.01
weight_decay = 0

# For deep4 they should be:
# lr = 1 * 0.01
# weight_decay = 0.5 * 0.001

batch_size = 64
n_epochs = 200
folds = range(5)
    
for group_id in groups:
    for i in folds:
        train_ids, val_ids, test_ids = spilt_dataset(subject_ids, i)
        X_train, y_train = load_data(train_ids, group_id, istraining = True)
        X_train, y_train = prepare_data(np.array(X_train), np.array(y_train).reshape(-1, 1))
        X_test, y_test = load_data(test_ids, group_id, istraining = False)
        X_test, y_test = prepare_data(np.array(X_test), np.array(y_test).reshape(-1, 1))
        X_val, y_val = load_data(val_ids, group_id, istraining = False)
        X_val, y_val = prepare_data(np.array(X_val), np.array(y_val).reshape(-1, 1))
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        y_val = y_val.squeeze()
        train_set = Dataset(X_train, y_train)
        valid_set = Dataset(X_val, y_val)
        cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
        device = 'cuda' if cuda else 'cpu'
        if cuda:
            torch.backends.cudnn.benchmark = True
        seed = 20200220  # random seed to make results reproducible
        # Set random seed to be able to reproduce results
        set_random_seeds(seed=seed, cuda=cuda)

        n_classes = 2
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
        
        cp = Checkpoint(dirname='Checkpoints', fn_prefix = 'SC-group{0}-{1}_'.format(group_id, i))
        early_stopping = EarlyStopping(patience=10)
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

        clf.fit(X= train_set, y=None, epochs=n_epochs)