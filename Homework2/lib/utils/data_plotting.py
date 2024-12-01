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

def get_labels(N = 7):
    if N == 7:
        labels = ["Sponginess", "Wonder level", "Crunchiness", 
                        "Loudness on impact", "Meme creativity", 
                        "Soap slipperiness", "Hype root"]
    elif N == 9:
        labels = ["Sponginess", "Wonder level", "Crunchiness", 
                        "Loudness on impact", "Meme creativity", 
                        "Soap slipperiness", "Hype root",
                        "Sin","Cos"]
    return labels

def plot_train_val_split(X_train_raw,X_val_raw, idx = 0, figsize=(14,5)):
    assert idx <= X_val_raw.shape[1]
    column_names = get_labels(X_train_raw.shape[1])
    N_train = X_train_raw.shape[0]
    N_val = X_val_raw.shape[0]
    x = np.arange(0,N_train)
    xx = np.arange(N_train,N_train+N_val)
    
    f,ax = plt.subplots(figsize=figsize)
    ax.plot(x, X_train_raw[:,idx], label="Train " + column_names[idx])
    ax.plot(xx, X_val_raw[:,idx], label="Val " + column_names[idx])
    ax.set_title('Train-Val Split')
    ax.legend()
    return f,ax

def plot_boxplot(X, figsize=(14,5)):
    column_names = get_labels(X.shape[1])
    df = pd.DataFrame(X, columns = column_names)
    
    f,ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df,ax = ax)
    plt.xticks(rotation=45)
    ax.set_title("Boxplot of data")
    plt.show()  
    return f,ax

def plot_normalized_data(X, norms, figsize=(14,5)):
    
    column_names = get_labels(X.shape[1])
    
    X_normalized = (X - norms[0]) / norms[1]
    
    # Convert to pandas
    df = pd.DataFrame(X, columns = column_names)
    df_normalized = pd.DataFrame(X_normalized, columns = column_names)
    df = df.melt(var_name='Column', value_name='Values')
    df_normalized = df_normalized.melt(var_name='Column', value_name='Values')
    
    
    f,ax = plt.subplots(1,2, figsize=figsize)
    sns.violinplot(x='Column', y='Values', data=df,
                   ax = ax[0])
    sns.violinplot(x='Column', y='Values', data=df_normalized,
                   ax = ax[1])
    ax[0].set_title("Original Data")
    ax[0].set_xticklabels(column_names, rotation=90)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('')
    ax[1].set_title("Data Normalized")
    ax[1].set_xticklabels(column_names, rotation=90)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    return f,ax

def plotData(X, N_train = None, figsize = (15,25)):
    column_names = get_labels(X.shape[1])
    col = ['r','b','g','m','k','y','c','hotpink','darkslategrey']
    f,ax = plt.subplots(X.shape[1],1, figsize=figsize)
    ax = ax.flatten()
    for i in range(X.shape[1]):
        ax[i].plot(X[:,i],color=col[i])
        if N_train is not None:
            ax[i].vlines(N_train,ymin=X[:,i].min(),ymax=X[:,i].max(),label="Train/Val Split")
            ax[i].legend()
        ax[i].set_title(column_names[i])
    return f,ax