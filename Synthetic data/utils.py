import os
import numpy as np
from rich.progress import track
from math import ceil
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.utils.class_weight import compute_class_weight 

def create_segments(start_index, run_length, seg_duration, overlap):
    '''
    input:
    start_index - the onset of the current run
    run_length - the number of samples in the run
    seg_duration - segment duration in seconds 
    overlap - float fraction of frame to overlap in range (0,1)
    
    output: a list of tuple ranges
    '''

    sampling_rate = 128
    end_index = start_index + run_length
    windows = []
    seg_length = int(sampling_rate * seg_duration)
    
    while start_index + seg_length <= end_index:
        windows.append([start_index, start_index + seg_length])
        start_index = start_index + seg_length - int(seg_length * overlap)
    return windows

def segment_X(data, istraining):
    '''Segment a run'''
    
    seg_duration = 2
    if istraining:
        overlap = 0.9
    else:
        overlap = 0
        
    start_pos = 0
    run_length = 4 * 128
    windows = create_segments(start_pos, run_length, seg_duration, overlap)
    # downsample each segement to 128Hz
#     x = [mne.filter.resample(data[:, windows[n][0]: windows[n][1]], down=1.25, npad='auto') 
#          for n in range(len(windows))]
    x = [data[windows[n][0]: windows[n][1],:]
         for n in range(len(windows))]    
    return x

def generate_data(subject_ids, iscrop, istraining):
    trials = []
    labels = []
    # for i in track(range(1, 1744), description = 'Processing...'):
    for i in track(subject_ids, description = 'Processing...'):
        path1 = '/home/henrywang/synthdata/weaker/class19001701/synthdata{0}.19001701.egg'.format(i)
        path2 = '/home/henrywang/synthdata/weaker/class19021701/synthdata{0}.19021701.egg'.format(i) 
        
        trial1 = []
        trial2 = []
        with open(path1, 'r') as file:
            for line in file:
                # read EEG data and save it into a list
                newline = [round(float(i), 3) for i in line.split()]
                trial1.append(newline)
        with open(path2, 'r') as file:
            for line in file:
                # read EEG data and save it into a list
                newline = [round(float(i), 3) for i in line.split()]
                trial2.append(newline) 
        trial1 = np.asarray(trial1)
        trial2 = np.asarray(trial2)
        if iscrop == True:
            x1 = segment_X(trial1, istraining)
            trials += x1
            labels += [0] * len(x1)
            x2 = segment_X(trial2, istraining)
            trials += x2
            labels += [1] * len(x2)
        else:
            trials.append(trial1)
            labels+=[0]
            trials.append(trial2)
            labels+=[1]

    X = np.asarray(trials)
    y = np.asarray(labels)
    return X, y

def prepare_data(X, y):
    
    # Z-score Normalization
    shape = X.shape
    for i in range(shape[0]):
        X[i,:, :] = scale(X[i,:, :])
        if (i+1)%int(shape[0]//10) == 0:
            print('{:.0%} done'.format((i+1)/shape[0]))
            
    # One-hot encoding y
    oh = OneHotEncoder()
    y = oh.fit_transform(y).toarray()
    
    return np.swapaxes(X, 1, 2), y

def onehot_labels(y):
    # One-hot encoding y
    oh = OneHotEncoder()
    y = oh.fit_transform(y).toarray()
    return y

def spilt_dataset(subject_ids, i):
    seeds = [42, 68, 188, 256, 1337]
    subject_ids = np.random.RandomState(seed=seeds[i]).permutation(subject_ids)
    ids_length = len(subject_ids)
    train_ids, val_ids, test_ids = np.split(subject_ids, [int(.4 * ids_length), int(.6 * ids_length)])
    return train_ids, val_ids, test_ids