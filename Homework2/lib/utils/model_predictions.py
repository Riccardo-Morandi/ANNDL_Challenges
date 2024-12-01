import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from scipy.stats.stats import pearsonr 

def get_sines(config):
    PERIOD = 96
    i = np.arange(0,PERIOD)
    time_arrays = np.zeros((PERIOD,2))    
    time_arrays[:,0] = np.sin(2*np.pi*i/PERIOD)
    time_arrays[:,1] = np.cos(2*np.pi*i/PERIOD)
    
    # Normalize 
    mean = config['norms'][0][-2:]
    std = config['norms'][1][-2:]
    
    time_arrays = (time_arrays - mean) / std
    
    return time_arrays

def fetch_next_sine_vals(idx, config):
    time_arrays = get_sines(config)
    
    if idx + config['telescope'] > 96:
        val1 = time_arrays[idx:]
        val2 = time_arrays[:(idx + config['telescope'])%96]
        vals = np.concatenate((val1,val2),axis=0)
    else:
        vals = time_arrays[idx:idx+config['telescope']]
    
    idx = idx + config["telescope"]
    if idx >= 96:
        idx = idx%96
    return vals, idx

def get_sine_location(XX, config):
    """ Try to find where in the sine the last value was
    """
    #TODO: hvad nu hvis window er mindre end 96?
    # Get the last values atleast
    XX = XX[-96:,-2:]

    time_arrays = get_sines(config) # (96,2)
    
    corr = np.zeros((96,))
    for i in range(96):
        val = np.roll(time_arrays,i,axis=0)
        corr[i],_ = pearsonr(XX[:,0],val[:,0])
    idx = np.argmax(corr)
    return 96-idx

def predict_training(model, X, config, N_predictions = 500, 
                     ASSISTED = True, post_processing=False):
    norms = config["norms"]
    
    # Normalize X with the same values used in training
    X_normalized = (X-norms[0]) / norms[1]
    
    # Check that N_predictions is valid
    N_train = X.shape[0] - int(config['val_split']*X.shape[0])
    assert N_predictions < N_train-config['window']-config["telescope"]
    if N_predictions%config["telescope"] != 0:
        N_predictions2 = N_predictions + config["telescope"]-N_predictions%config["telescope"]
    else:
        N_predictions2 = N_predictions
        
    # Maybe add a line here to guard against overflow?
    CONST = 40
    len_past = int(config['window']*CONST)
    past = X_normalized[:len_past]    
    
    # X_data = X_normalized[:N_predictions]
    # labels = X_normalized[config['window']:N_predictions+config['window']]
    start = len_past-config['window']
    X_data = X_normalized[start:start+N_predictions2]
    labels = X_normalized[len_past:len_past+N_predictions2,:7]

    # # Get windowed data
    # input_data = X_normalized[len_past-config['window']:]
    # X_window, y_window = array_to_timeseries(input_data, config, 
    #                                           N_predictions2)

    # if ASSISTED:
    #     # Make prediction
    #     pred = model.predict(X_window)
        
    #     # Reshape
    #     pred = np.reshape(pred,(-1,7))
    #     pred = pred[:N_predictions]
    #     # pred = np.reshape(pred,(N_predictions,7))
    # else:        
    # Initialize arrays
    output_list = []
    X_temp = X_data[:config["window"]][None,:]

    # Find out where in the sinus-the last sample was
    idx_next = get_sine_location(X_temp[0,:,:], config)

    for i in tqdm(range(0,N_predictions2,config["telescope"])):            
        # Get model output
        model_out = model.predict(X_temp)
        
        # Append result
        output_list.append(model_out[0])
        
        if ASSISTED:
            # Get the next value in list
            correct_result = labels[i:i+config["telescope"]][None,:]
            
            # Append the correct sine/cosine to the model
            if config['add_features']:
                vals, idx_next = fetch_next_sine_vals(idx_next, config)
                correct_result = np.concatenate((correct_result,vals[None,:]),axis=2)
                
            X_temp = tf.concat((X_temp[:,config["telescope"]:,:],correct_result), axis=1)
        else:
            if config['add_features']:
                vals, idx_next = fetch_next_sine_vals(idx_next, config)
                next_value = np.concatenate((output_list[-1][None,:],vals[None,:]),axis=2)
            else:
                next_value = output_list[-1][None,:]
            
            X_temp = tf.concat((X_temp[:,config["telescope"]:,:],next_value), axis=1)

    pred_tf = tf.stack(output_list)
    pred = np.array(pred_tf) # Overcome an assign error
    
    # Reshape 
    pred_tf = tf.reshape(pred_tf,(-1,7))
    pred = np.reshape(pred,(-1,7))
    pred_tf = pred_tf[:N_predictions]
    pred = pred[:N_predictions]
    # pred_tf = tf.reshape(pred_tf,(N_predictions,7))
    # pred = np.reshape(pred,(N_predictions,7))
        
    # label = np.reshape(y_window,(-1,7))
    # label = label[:N_predictions]
    # # label = np.reshape(y_window,(N_predictions,7))
    # labels = labels[:N_predictions]
    # assert np.all(label==labels)
    label = labels

    # Insert your postprocessing here
    if post_processing:
        past = (past * norms[1]) + norms[0]
        pred = (pred * norms[1][:7]) + norms[0][:7]   
        label = (label * norms[1][:7]) + norms[0][:7] 
    return past, pred, label

def predict_validation(model, X, config, N_predictions = 500, 
                     ASSISTED = True, post_processing=False):
    norms = config["norms"]
    
    # Normalize X with the same values used in training
    X_normalized = (X-norms[0]) / norms[1]
    
    # Check that N_predictions is valid
    N_train = X.shape[0] - int(config['val_split']*X.shape[0])
    N_val = int(config['val_split']*X.shape[0])
    assert N_predictions < N_val
    if N_predictions%config["telescope"] != 0:
        N_predictions2 = N_predictions + config["telescope"]-N_predictions%config["telescope"]
    else:
        N_predictions2 = N_predictions
        
    X_data = X_normalized[N_train-config['window']:N_train+N_predictions2-config['window']]
    labels = X_normalized[N_train:N_train+N_predictions2,:7]
    
    
    CONST = 40
    len_past = int(config['window']*CONST)
    past = X_normalized[N_train-len_past:N_train]


    # # Get windowed data
    # input_data = X_normalized[N_train-config['window']:]
    # X_window, y_window = array_to_timeseries(input_data, config, 
    #                                          N_predictions2)
    
    # if ASSISTED:
    #     # Make prediction
    #     pred = model.predict(X_window)
        
    #     # Reshape
    #     pred = np.reshape(pred,(-1,7))
    #     pred = pred[:N_predictions]
    #     # pred = np.reshape(pred,(N_predictions,7))
    # else:
    # Initialize arrays
    output_list = []
    X_temp = X_data[:config["window"]][None,:]
    
    # Find out where in the sinus-the last sample was
    idx_next = get_sine_location(X_temp[0,:,:], config)

    for i in tqdm(range(0,N_predictions2,config["telescope"])):
        model_out = model.predict(X_temp)
        
        # Append result
        output_list.append(model_out[0])
        
        if ASSISTED:
            # Get the correct next value in the list
            correct_result = labels[i:i+config["telescope"]][None,:]
            
            # Append the correct sine/cosine to the model
            if config['add_features']:
                vals, idx_next = fetch_next_sine_vals(idx_next, config)
                correct_result = np.concatenate((correct_result,vals[None,:]),axis=2)

            X_temp = tf.concat((X_temp[:,config["telescope"]:,:],correct_result), axis=1)
        else:
            if config['add_features']:
                vals, idx_next = fetch_next_sine_vals(idx_next, config)
                next_value = np.concatenate((output_list[-1][None,:],vals[None,:]),axis=2)
            else:
                next_value = output_list[-1][None,:]
                
            X_temp = tf.concat((X_temp[:,config["telescope"]:,:],next_value), axis=1)

    pred_tf = tf.stack(output_list)
    pred = np.array(pred_tf) # Overcome an assign error
    
    # Reshape
    pred_tf = tf.reshape(pred_tf,(-1,7))
    pred_tf = pred_tf[:N_predictions]
    pred = np.reshape(pred,(-1,7))
    pred = pred[:N_predictions]
    # pred_tf = tf.reshape(pred_tf,(N_predictions,7))
    # pred = np.reshape(pred,(N_predictions,7))
        
    # label = np.reshape(y_window,(-1,7))
    # label = label[:N_predictions]
    # # label = np.reshape(y_window,(N_predictions,7))
    # labels = labels[:N_predictions]
    # assert np.all(label==labels)
    label = labels

    # Insert your postprocessing here
    if post_processing:
        past = (past * norms[1]) + norms[0]
        pred = (pred * norms[1][:7]) + norms[0][:7]
        label = (label * norms[1][:7]) + norms[0][:7]
    return past, pred, label

def predict_future(model, X, config, 
                   N_predictions = 500, 
                   post_processing=False):
    norms = config["norms"]
    
    # Normalize X with the same values used in training
    X_normalized = (X-norms[0]) / norms[1]
    
    # Check that N_predictions is valid
    if N_predictions%config["telescope"] != 0:
        # N_predictions += N_predictions%config["telescope"]
        N_predictions2 = N_predictions + config["telescope"]-N_predictions%config["telescope"]
    else:
        N_predictions2 = N_predictions
        
    X_data = X_normalized[-config['window']:]
    
    # Initialize arrays
    output_list = []
    X_temp = X_data[None,:]
    
    # Find out where in the sinus-the last sample was
    if config['add_features']:
        idx_next = get_sine_location(X_temp[0,:,:], config)
    
    for i in tqdm(range(0,N_predictions2,config["telescope"])):
        
        model_out = model.predict(X_temp)
        output_list.append(model_out[0])
        
        if config['add_features']:
            vals, idx_next = fetch_next_sine_vals(idx_next, config)
            next_value = np.concatenate((output_list[-1][None,:],vals[None,:]),axis=2)
        else:
            next_value = output_list[-1][None,:]
        
        X_temp = tf.concat((X_temp[:,config["telescope"]:,:],next_value), axis=1)

    pred_tf = tf.stack(output_list)
    pred = np.array(pred_tf) # Overcome an assign error
    
    # Reshape
    pred_tf = tf.reshape(pred_tf,(-1,7))
    pred = np.reshape(pred,(-1,7))
    pred_tf = pred_tf[:N_predictions]
    pred = pred[:N_predictions]
    # pred_tf = tf.reshape(pred_tf,(N_predictions,7))
    # pred = np.reshape(pred,(N_predictions,7))

    # Insert your postprocessing here
    past = X_normalized # All the data before
    if post_processing:
        past = (past * norms[1]) + norms[0] 
        pred = (pred * norms[1][:7]) + norms[0][:7]  
    return past, pred

def array_to_timeseries(X, config, N_predictions = 500):
    """
    split = ["val",N_train] # first element be either "train" or "val"
    """
    
    window = config["window"]
    stride = config["telescope"]
    telescope = config["telescope"]
    
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []

    temp_X = np.copy(X)
    temp_labels = np.copy(X[:,:7])
        
    for idx in np.arange(0,N_predictions,stride):
        dataset.append(temp_X[idx:idx+window])
        labels.append(temp_labels[idx+window:idx+window+telescope])


    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels
