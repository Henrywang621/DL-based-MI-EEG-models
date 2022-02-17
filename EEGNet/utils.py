import os
import numpy as np
import mne
from rich.progress import track
from mne import pick_types
from mne.io import read_raw_edf
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

    sampling_rate = 160
    end_index = start_index + run_length
    windows = []
    seg_length = int(sampling_rate * seg_duration)
    
    while start_index + seg_length <= end_index:
        windows.append([start_index, start_index + seg_length])
        start_index = start_index + seg_length - int(seg_length * overlap)
    return windows

def segment_X(event, data, istraining):
    '''Segment a run'''
    
    sampling_rate = 160
    seg_duration = 2
    if istraining:
        overlap = 0
    else:
        overlap = 0
        
    start_pos = ceil(float(event[0]) * sampling_rate)
    run_length = ceil(float(event[1]) * sampling_rate)
    windows = create_segments(start_pos, run_length, seg_duration, overlap)
    # downsample each segement to 128Hz
    x = [mne.filter.resample(data[:, windows[n][0]: windows[n][1]], down=1.25, npad='auto') 
         for n in range(len(windows))]
    return x

def read_edffile(subject_id, current_run):
    base_path = '/home/henrywang/PythonPractice/testCP_RCNN/datasets/BCI2000/S{0:03}/S{1:03}R{2:02}.edf'
    path = base_path.format(subject_id, subject_id, current_run)
    raw = read_raw_edf(path, preload=True, verbose=False)
    onset = raw.annotations.onset
    duration = raw.annotations.duration
    description = raw.annotations.description
    # events[Oneset of the event in seconds, Duration of the event in seconds, Description]
    events = [[onset[i], duration[i], description[i]] for i in range(len(raw.annotations))]
    picks = pick_types(raw.info, eeg=True)
    raw.filter(l_freq=4, h_freq=None, picks=picks)
    data = raw.get_data(picks=picks)
    
    return events, data

def generate_data(event, data,  istraining, label):
    x = segment_X(event, data, istraining)
    y = [0] if label == 0 else [1]
    y *= len(x)
    return x, y
    
    
'''
Experimental groups of Binary classification

Group1: Left hand & Right hand -> T1-run_type2 (label:0) & T2-run_type2 (label:1);
Group2: Left hand & Both feet -> T1-run_type2 (label:0) & T2-run_type3 (label:1);
Group3: Right hand & Both feet -> T2-run_type2 (label:0) & T2-run_type3 (label:1);
Group4: Rest & Left Hand -> T0 (label:0) & T1-run_type2 (label:1);
Group5: Rest & Right Hand -> T0 (label:0) & T2-run_type2 (label:1);
Group6: Rest & Both Feet -> T0 (label:0) & T2-run_type3 (label:1);
Group7: Rest & Both hands -> T0 (label:0) & T1-run_type3 (label:1);
Group8: Both hands & Both Feet -> T1-run_type3 (label:0) & T2-run_type3 (label:1);
Group9: Right hand & Both hands -> T2-run_type2 (label:0) & T1-run_type3 (label:1);
Group10: Left hand & Both hands -> T1-run_type2 (label:0) & T1-run_type3 (label:1); 
'''

def load_data(subject_ids, group_id, istraining):
    
    crops = []
    labels = []
    runs_type1 = [2]
    runs_type2 = [4, 8, 12]
    runs_type3 = [6, 10, 14]
    MI_runs = [2, 4, 6, 8, 10, 12, 14] 
    
    if group_id == 1:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in runs_type2:
                events, data = read_edffile(subject_id, MI_run)
                for event in events[:-1]:
                    if event[2] == 'T0':
                        continue
                    else:
                        x = segment_X(event, data, istraining)
                        y = [0] if event[2] == 'T1' else [1]
                        y *= len(x)
                        crops += x
                        labels += y
                        
    elif group_id == 2:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in runs_type2 + runs_type3:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if MI_run in runs_type2 and event[2] == 'T1':
                        x, y = generate_data(event, data, istraining, label = 0)
                        crops += x
                        labels += y
                    elif MI_run in runs_type3 and event[2] == 'T2':
                        x, y = generate_data(event, data, istraining, label = 1)
                        crops += x
                        labels += y
    
    elif group_id == 3:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in runs_type2 + runs_type3:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if event[2] == 'T2':
                        if MI_run in runs_type2:
                            x, y = generate_data(event, data, istraining, label = 0)
                            crops += x
                            labels += y
                        elif MI_run in runs_type3:
                            x, y = generate_data(event, data, istraining, label = 1)
                            crops += x
                            labels += y
                            
    elif group_id == 4:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in MI_runs:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if event[2] == 'T0':
                        x, y = generate_data(event, data, istraining, label = 0)
                        crops += x
                        labels += y
                    elif MI_run in runs_type2 and event[2] == 'T1':
                        x, y = generate_data(event, data, istraining, label = 1)
                        crops += x
                        labels += y

    elif group_id == 5:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in MI_runs:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if event[2] == 'T0':
                        x, y = generate_data(event, data, istraining, label = 0)
                        crops += x
                        labels += y
                    elif MI_run in runs_type2 and event[2] == 'T2': 
                        x, y = generate_data(event, data, istraining, label = 1)
                        crops += x
                        labels += y
                        
    elif group_id == 6:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in MI_runs:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if event[2] == 'T0':
                        x, y = generate_data(event, data, istraining, label = 0)
                        crops += x
                        labels += y
                    elif MI_run in runs_type3 and event[2] == 'T2': 
                        x, y = generate_data(event, data, istraining, label = 1)
                        crops += x
                        labels += y

    elif group_id == 7:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in MI_runs:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events:
                        if event[2] == 'T0':
                            x, y = generate_data(event, data, istraining, label = 0)
                            crops += x
                            labels += y
                        elif MI_run in runs_type3 and event[2] == 'T1':
                            x, y = generate_data(event, data, istraining, label = 1)
                            crops += x
                            labels += y
    
    elif group_id == 8:
        for subject_id in track(subject_ids, description = 'Processing...'):        
            for MI_run in runs_type3:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:        
                    if event[2] == 'T0':
                        continue
                    else:
                        x = segment_X(event, data, istraining)
                        y = [0] if event[2] == 'T1' else [1]
                        y *= len(x)
                        crops += x
                        labels += y

    elif group_id == 9:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in runs_type2 + runs_type3:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if MI_run in runs_type2 and event[2] == 'T2':
                        x, y = generate_data(event, data, istraining, label = 0)
                        crops += x
                        labels += y
                    elif MI_run in runs_type3 and event[2] == 'T1':
                        x, y = generate_data(event, data, istraining, label = 1)
                        crops += x
                        labels += y   

    elif group_id == 10:
        for subject_id in track(subject_ids, description = 'Processing...'):
            for MI_run in runs_type2 + runs_type3:
                events, data = read_edffile(subject_id, MI_run)
                for event in events:
                    if event[2] == 'T1':
                        if MI_run in runs_type2:
                            x, y = generate_data(event, data, istraining, label = 0)
                            crops += x
                            labels += y
                        elif MI_run in runs_type3:
                            x, y = generate_data(event, data, istraining, label = 1)
                            crops += x
                            labels += y        
                        
    else:
        raise ValueError('group_id should be in the range of 1 to 6 !!!')
    
    return crops, labels 

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
    
    return X, y

def get_weights(y):
    y_integers = np.argmax(y, axis=1)
    classes = np.unique(y_integers)
    class_weights = compute_class_weight("balanced", classes = classes, y = y_integers)
    class_weights = [round(weight, 2) for weight in class_weights]
    d_class_weights = dict(zip(classes, class_weights))
    return d_class_weights

def spilt_dataset(subject_ids, i):
    seeds = [42, 68, 188, 256, 1337]
    subject_ids = np.random.RandomState(seed=seeds[i]).permutation(subject_ids)
    ids_length = len(subject_ids)
    train_ids, val_ids, test_ids = np.split(subject_ids, [int(.7 * ids_length), int(.8 * ids_length)])
    return train_ids, val_ids, test_ids