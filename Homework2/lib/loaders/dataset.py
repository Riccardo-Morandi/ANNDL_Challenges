import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
tfkl = tf.keras.layers

#%% Data functions
def get_labels():
    labels = ["Sponginess", "Wonder level", "Crunchiness", 
                    "Loudness on impact", "Meme creativity", 
                    "Soap slipperiness", "Hype root"]
    return labels
def get_loaders(X_train, y_train, X_val, y_val,config):
    
    batch_size = config["batch_size"]
    
    # Convert to tensorflow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Shuffle into correct batch sizes
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset

def get_datasets(config, 
                 data_path = '../../data/Training.csv'):
    # Load data
    X = get_raw_data(data_path)
    
    # Add sine and cosine to dataset
    if config["add_features"] == True:
        X = add_features(X)
    
    # Split data
    X_train_raw, X_val_raw, N_train =  split_raw_data(X, config)
    
    # Normalize
    X_train_raw, X_val_raw, norms = normalize_raw_data(X_train_raw,
                                                       X_val_raw,
                                                       config)
    
    # Perform windowing    
    X_train, y_train = build_sequences(X_train_raw, config)
    X_val, y_val = build_sequences(X_val_raw, config)
    
    return X, X_train_raw, X_val_raw, X_train, y_train, X_val, y_val, norms 

def build_sequences(X, config):
    window = config["window"]
    stride = config["stride"]
    telescope = config["telescope"]
    
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    
    temp_X = np.copy(X)
    temp_label = np.copy(X[:,:7])
    padding_len = X.shape[0]%window

    if(padding_len != 0):
        # Compute padding length
        padding_len = window - X.shape[0]%window
        padding = np.zeros((padding_len,temp_X.shape[1]), dtype='float64')
        temp_X = np.concatenate((padding,X))
        padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding,temp_label))
        assert len(temp_X) % window == 0

    for idx in np.arange(0,temp_X.shape[0]-window-telescope,stride):
        dataset.append(temp_X[idx:idx+window])
        labels.append(temp_label[idx+window:idx+window+telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def normalize_raw_data(X_train_raw, X_val_raw , config):
    """
    mode = 0 : nomalize to zero mean and unit variance
    mode = 1 : normalize to [0,1]
    """
    mode = config["normalization_mode"]
    
    norms = []
    if mode == 0:
        train_mean = X_train_raw.mean(axis=0)
        train_std = X_train_raw.std(axis=0)
        norms.append(train_mean)
        norms.append(train_std)
        
        X_train_raw = (X_train_raw - train_mean) / train_std
        X_val_raw = (X_val_raw - train_mean) / train_std
    elif mode == 1:
        train_min = X_train_raw.min(axis=0)
        train_max = X_train_raw.max(axis=0)
        norms.append(train_min)
        norms.append((train_max - train_min))
        
        X_train_raw = (X_train_raw - train_min) / (train_max - train_min)
        X_val_raw = (X_val_raw - train_min) / (train_max - train_min)
    else:
        raise ValueError("Wrong normalization mode")
    return X_train_raw, X_val_raw, norms

def split_raw_data(X, config):
    val_split = config["val_split"]
    val_size = int(X.shape[0]*val_split)
    N_train = X.shape[0] - val_size
    
    X_train_raw = X[:-val_size]
    X_val_raw = X[-val_size:]
    return X_train_raw, X_val_raw, N_train

def get_raw_data(data_path = '../../data/Training.csv'):
    column_names = get_labels()

    # Read data
    df = pd.read_csv(data_path, header=0, names=column_names)
    df.dropna(axis=0, how='any', inplace=True)
    X = df.to_numpy().astype("float32")
    return X

def add_features(X):
    PERIOD = 96
    i = np.arange(0,X.shape[0])
    
    time_arrays = np.zeros((X.shape[0],2),dtype=np.float32)    
    time_arrays[:,0] = np.sin(2*np.pi*i/PERIOD)
    time_arrays[:,1] = np.cos(2*np.pi*i/PERIOD)
    
    X_new = np.concatenate((X,time_arrays),axis=1)
    return X_new
#%%
if __name__ == '__main__':
    sys.path.append("..")
    import utils as util
    
    # Set up paths
    data_path = '../../data/Training.csv'
    image_path = "../../figures/" # Where to save images
    
    # Set up config
    config = {
        "batch_size" : 32,
        "val_split" : 0.1,
        "add_features": True,
        # Parameters for windowing
        "window" : 200,
        "stride" : 10,
        "telescope" : 96,
        # Parameters for data normalization
        "normalization_mode" : 0
        }    
    X = get_raw_data(data_path)
    
    X = add_features(X)
    
    # f,ax = util.plot_boxplot(X)
    # plt.show()
    
    # Split data
    X_train_raw, X_val_raw, N_train =  split_raw_data(X, config)
    
    # f,ax = util.plotData(X, N_train, figsize=(15,27))
    # plt.show()
    # f.savefig(image_path+"original_data.png",bbox_inches='tight',dpi=300)
    
    # f,ax = util.plot_train_val_split(X_train_raw,X_val_raw, idx = 0, figsize=(14,5))
    # plt.show()
    # f.savefig(image_path+"train_val_split.png",bbox_inches='tight',dpi=300)
    
    # Normalize
    X_train_raw, X_val_raw, norms = normalize_raw_data(X_train_raw,X_val_raw,config)
    
    # f,ax = util.plot_normalized_data(X, norms, figsize=(14,5))
    # plt.show()
    # f.savefig(image_path+"data_normalization.png",bbox_inches='tight',dpi=300)

    
    X_train, y_train = build_sequences(X_train_raw, config)
    X_val, y_val = build_sequences(X_val_raw, config)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    # Get data loaders
    batch_size = 32
    train_dataset, val_dataset = get_loaders(X_train, 
                                             y_train, 
                                             X_val, 
                                             y_val,
                                             config)

    
    
    
    