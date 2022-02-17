import os
import numpy as np
import mne
from rich.progress import track
from mne import pick_types
from mne.io import read_raw_edf
from math import ceil
from sklearn.preprocessing import OneHotEncoder, scale

def create_trial(start_index, run_length, trial_duration):
    '''
    input:
    start_index - the onset of the current run
    run_length - the number of samples in the run
    trial_duration - the constructed trial in seconds 
    
    output: a list of tuple ranges
    '''

    sampling_rate = 160
    trial_length = int(trial_duration * sampling_rate)
    pad = (trial_length - run_length) // 2
    reminder = (trial_length - run_length) % 2
    end_index = start_index + run_length
    windows = []
    windows += [round(start_index - pad - reminder), round(end_index + pad)]
    return windows

def construct_X(event, data):
    '''Segment a run'''
    
    sampling_rate = 160
    trial_duration = 6
    start_pos = int(event[0] * sampling_rate)
    run_length = int(event[1] * sampling_rate)
    windows = create_trial(start_pos, run_length, trial_duration)
    # downsample each segement to 128Hz
    x = data[:, windows[0]: windows[1]]
#     x = mne.filter.resample(data[:, windows[0]: windows[1]], down=1.25, npad='auto')
    return x

def construct_Xr(event, data):
    # To generate the same data every time
    random_seeds = [42, 1, 1120, 150, 256, 303, 80, 5374, 646, 763, 4763, 
                    947, 1004, 7, 1157, 1234, 6402, 1314, 1337, 1448, 
                    662]
    trial_duration = 6
    sampling_rate = 160
    x = []
    y = []
    trial_length = int(trial_duration * sampling_rate)
    upper_limit = int(event[0][1] * sampling_rate) - trial_length
    for i in range(len(random_seeds)):
        offset = np.random.RandomState(seed=random_seeds[i]).randint(0, upper_limit)
#         x.append(data[:, offset:offset + trial_length])
        x.append(mne.filter.resample(data[:, offset:offset + trial_length], 
                                     down=1.25, npad='auto'))
        y += [0]
    return x, y

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

def load_data(subject_ids, group_id, istwoclasses = True):
    
    trials = []
    labels = []
    runs_type1 = [2]
    runs_type2 = [4, 8, 12]
    runs_type3 = [6, 10, 14]
    MI_runs = [2, 4, 6, 8, 10, 12, 14] 
    
    if istwoclasses == True:
        if group_id == 1:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type2:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if event[2] == 'T0':
                            continue
                        else:
                            x = construct_X(event, data)
                            y = [0] if event[2] == 'T1' else [1]
                            trials.append(x)
                            labels += y
        
        elif group_id == 2:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type2 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if MI_run in runs_type2 and event[2] == 'T1':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [0]
                        elif MI_run in runs_type3 and event[2] == 'T2':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [1]    
                            
        elif group_id == 3:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type2 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if event[2] == 'T2':
                            if MI_run in runs_type2:
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [0]
                            elif MI_run in runs_type3:
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]
                                
        elif group_id == 4:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type1 + runs_type2:
                    events, data = read_edffile(subject_id, MI_run)
                    if MI_run in runs_type1:
                        x, y = construct_Xr(events, data)
                        trials += x
                        labels += y
                    else:
                        for event in events[:-1]:
                            if event[2] == 'T1':
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]
                                
        elif group_id == 5:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type1 + runs_type2:
                    events, data = read_edffile(subject_id, MI_run)
                    if MI_run in runs_type1:
                        x, y = construct_Xr(events, data)
                        trials += x
                        labels += y
                    else:
                        for event in events[:-1]:
                            if event[2] == 'T2':
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]

        elif group_id == 6:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type1 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    if MI_run in runs_type1:
                        x, y = construct_Xr(events, data)
                        trials += x
                        labels += y
                    else:
                        for event in events[:-1]:
                            if event[2] == 'T2':
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]
        
        elif group_id == 7:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type1 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    if MI_run in runs_type1:
                        x, y = construct_Xr(events, data)
                        trials += x
                        labels += y
                    else:
                        for event in events[:-1]:
                            if event[2] == 'T1':
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]     

        if group_id == 8:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if event[2] == 'T1':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [0]
                        elif event[2] == 'T2':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [1]                               

        elif group_id == 9:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type2 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if MI_run in runs_type2 and event[2] == 'T2':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [0]
                        elif MI_run in runs_type3 and event[2] == 'T1':
                            x = construct_X(event, data)
                            trials.append(x)
                            labels += [1]
                            
        elif group_id == 10:
            for subject_id in track(subject_ids, description = 'Processing...'):
                for MI_run in runs_type2 + runs_type3:
                    events, data = read_edffile(subject_id, MI_run)
                    for event in events[:-1]:
                        if event[2] == 'T1':
                            if MI_run in runs_type2:
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [0]
                            elif MI_run in runs_type3:
                                x = construct_X(event, data)
                                trials.append(x)
                                labels += [1]                            
                            
    else:
        pass

    
    
    return trials, labels

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

def spilt_dataset(subject_ids, i):
    seeds = [42, 68, 188, 256, 1337]
    subject_ids = np.random.RandomState(seed=seeds[i]).permutation(subject_ids)
    ids_length = len(subject_ids)
    train_ids, val_ids, test_ids = np.split(subject_ids, [int(.7 * ids_length), int(.8 * ids_length)])
    return train_ids, val_ids, test_ids