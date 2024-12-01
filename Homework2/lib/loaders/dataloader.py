import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tfk = tf.keras
tfkl = tf.keras.layers

def get_labels():
    labels = ["Sponginess", "Wonder level", "Crunchiness", 
                    "Loudness on impact", "Meme creativity", 
                    "Soap slipperiness", "Hype root"]
    return labels

def get_loaders(data_path = '../../data/Training.csv',  
                val_split = 0.1,
                window=200, 
                stride=20, 
                telescope=100,
                normalization_mode = 0):
    # Get label names
    target_labels = get_labels()

    # Read data
    df = pd.read_csv(data_path, header=0, names=target_labels)
    df.dropna(axis=0, how='any', inplace=True)

    train_df,val_df = split_raw_data(df, val_split = val_split)
    train_df,val_df,norms = normalize_raw_data(train_df, val_df , mode = normalization_mode)

    X_train, y_train = build_sequences(train_df, target_labels, window, stride, telescope)
    X_val, y_val = build_sequences(val_df, target_labels, window, stride, telescope)
    return  X_train, y_train, X_val, y_val, norms

def get_tf_datasets(X_train, y_train, X_val, y_val,
                    batch_size = 32):
    # Convert to tensorflow Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Shuffle into correct batch sizes
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    
    return train_dataset, val_dataset

def build_sequences(df, target_labels, window=200, stride=20, telescope=100):
    # Sanity check to avoid runtime errors
    assert window % stride == 0
    dataset = []
    labels = []
    temp_df = df.copy().values
    temp_label = df[target_labels].copy().values
    padding_len = len(df)%window

    if(padding_len != 0):
        # Compute padding length
        padding_len = window - len(df)%window
        padding = np.zeros((padding_len,temp_df.shape[1]), dtype='float64')
        temp_df = np.concatenate((padding,df))
        padding = np.zeros((padding_len,temp_label.shape[1]), dtype='float64')
        temp_label = np.concatenate((padding,temp_label))
        assert len(temp_df) % window == 0

    for idx in np.arange(0,len(temp_df)-window-telescope,stride):
        dataset.append(temp_df[idx:idx+window])
        labels.append(temp_label[idx+window:idx+window+telescope])

    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels

def plotData(X, column_names):
    col = ['r','b','g','m','k','y','c']
    f,ax = plt.subplots(X.shape[1],1, figsize=(15,25))
    ax = ax.flatten()
    for i in range(X.shape[1]):
        ax[i].plot(X[:,i],color=col[i])
        ax[i].set_title(column_names[i])
    plt.show()
    return None
def plot_train_val_split(X_train_raw,X_val_raw, idx = 0):
    f,ax = plt.subplots(figsize=(14,5))
    ax.plot(X_train_raw[column_names[idx]], label="Train " + column_names[idx])
    ax.plot(X_val_raw[column_names[idx]], label="Val " + column_names[idx])
    ax.set_title('Train-Val Split')
    ax.legend()
    return f,ax
def split_raw_data(df, val_split = 0.1):
    val_size = int(df.shape[0]*val_split)
    
    train_df = df.iloc[:-val_size]
    val_df = df.iloc[-val_size:]
    return train_df, val_df

def normalize_raw_data(train_df, val_df , mode = 0):
    """
    mode = 0 : nomalize to zero mean and unit variance
    mode = 1 : normalize to [0,1]
    """
    norms = []
    if mode == 0:
        train_mean = train_df.mean()
        train_std = train_df.std()
        norms.append(train_mean)
        norms.append(train_std)
        
        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
    elif mode == 1:
        train_min = train_df.min()
        train_max = train_df.max()
        norms.append(train_min)
        norms.append((train_max - train_min))
        
        train_df = (train_df - train_min) / (train_max - train_min)
        val_df = (val_df - train_min) / (train_max - train_min)
    else:
        raise ValueError("Wrong normalization mode")
    return train_df, val_df, norms

def plot_normalized_data(df, norms):
    df_normed = (df - norms[0]) / norms[1]
    df_normed = df_normed.melt(var_name='Column', value_name='Values')
    
    df_copy = df
    df_copy = df_copy.melt(var_name='Column', value_name='Values')
    
    f,ax = plt.subplots(1,2, figsize=(14,5))
    sns.violinplot(x='Column', y='Values', data=df_copy,
                   ax = ax[0])
    sns.violinplot(x='Column', y='Values', data=df_normed,
                   ax = ax[1])
    ax[0].set_title("Original Data")
    ax[0].set_xticklabels(df.keys(), rotation=90)
    ax[1].set_title("Data Normalized")
    ax[1].set_xticklabels(df.keys(), rotation=90)
    return f,ax
def get_raw_data(data_path = '../../data/Training.csv'):
    column_names = get_labels()

    # Read data
    df = pd.read_csv(data_path, header=0, names=column_names)
    df.dropna(axis=0, how='any', inplace=True)
    X = df.to_numpy().astype("float32")
    return X

#%%
if __name__ == '__main__':
    data_path = '../../data/Training.csv'
    # Get label names
    column_names = get_labels()
    # Read data
    df = pd.read_csv(data_path, header=0, names=column_names)
    df.dropna(axis=0, how='any', inplace=True)
    
    #%% Split into train and val
    val_split = 0.2
    train_df,val_df = split_raw_data(df, val_split = val_split)
    print("Train shape:",train_df.shape,"\nVal shape  :", val_df.shape)
    
    # Normalize both features and labels   
    train_df,val_df,norms = normalize_raw_data(train_df, val_df , mode = 0)
    
    f,ax = plot_train_val_split(train_df,val_df, idx = 1)
    plt.show()
    
    # Plot normalization
    f,ax = plot_normalized_data(df, norms)
    plt.show()       
            
    #%%
    # Parameters for windowing
    window = 300
    stride = 2
    target_labels = df.columns
    telescope = 1
    
    X_train, y_train = build_sequences(train_df, target_labels, window, stride, telescope)
    X_val, y_val = build_sequences(val_df, target_labels, window, stride, telescope)
    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    #%% Convert to tf.dataset 
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    
    batch_size = 32
    SHUFFLE_BUFFER_SIZE = 100
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    
    
    
    
    