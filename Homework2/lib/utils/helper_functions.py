import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .model_predictions import predict_training, predict_validation
from .model_predictions import predict_future


def set_seed(seed=12):
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return None


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


def plotTraining(config):
    best_epoch = np.argmin(config['val_loss'])
    plt.figure(figsize=(12,4))
    plt.plot(config['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(config['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.9, ls='--', color='r')
    plt.title('Mean Squared Error (Loss)')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
    
    if "mae" in config:
        plt.figure(figsize=(12,4))
        plt.plot(config['mae'], label='Training accuracy', alpha=.8, color='#ff7f0e')
        plt.plot(config['val_mae'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
        plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
        plt.title('Mean Absolute Error')
        plt.legend()
        plt.grid(alpha=.3)
        plt.show()
    
    plt.figure(figsize=(12,3))
    plt.plot(config['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.9, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()
    return None

def show_predictions(model,X, config, 
                     mode = "train",
                     N_predictions = 500, 
                     ASSISTED = True, 
                     post_processing=True,
                     figsize=(13,11)):
    """
    mode = "train", "val", "future"
    """
    columns = get_labels(X.shape[1])
    
    # Make predictions
    if mode == "train":
        past, pred, label = predict_training(model, X, config,
                                             N_predictions,
                                             ASSISTED, 
                                             post_processing)
        mse = np.mean((pred-label)**2)
        rmse = np.sqrt(mse)
        
        # Only show the past N dots
        # N = int(config["window"]*1.5)
        N = int(N_predictions/3)
        past = past[-N:]
    elif mode == "val":
        past, pred, label = predict_validation(model, X, config, 
                                               N_predictions, 
                                               ASSISTED, 
                                               post_processing)
        mse = np.mean((pred-label)**2)
        rmse = np.sqrt(mse)
        # N = int(config["window"]*1.5)
        N = int(N_predictions/3)
        past = past[-N:]
    elif mode == "future":
        past, pred = predict_future(model, X, config,N_predictions,
                                     post_processing)
        # Only show the past N dots
        # N = int(config["window"]*1.5)
        N = int(N_predictions/3)
        past = past[-N:]
    else:
        raise ValueError("Wrong mode selected should be 'train','val' or 'future'")

    c = ["b", "seagreen", "maroon"]
    
    x_past = np.arange(0,past.shape[0])
    x_future = np.arange(past.shape[0],past.shape[0]+pred.shape[0])
            
    # f,ax = plt.subplots(7,1, figsize=(13,17))
    f,ax = plt.subplots(4,2, figsize=figsize)
    f.tight_layout(rect=[0, 0.03, 1, 0.96])
    ax = ax.flatten()
    for i in range(7):
        ax[i].plot(x_past, past[:,i],'-', alpha=0.8, color=c[0], label="Past")
        if mode != "future":
            ax[i].plot(x_future, label[:,i],'-', color=c[1], label="Label")
        ax[i].plot(x_future, pred[:,i],'.', color=c[2], label="Prediction")
        ax[i].set_title(columns[i])
        ax[i].legend()
    for j in range(i+1,len(ax)):
        ax[j].set_visible(False)
    
    if mode == "train":
        f.suptitle("Prediction on the training set rmse: "+str(round(rmse,3)))
    elif mode == "val":
        f.suptitle("Prediction on the validation set rmse: "+str(round(rmse,3)))
    elif mode == "future":
        f.suptitle("Prediction of the future")    
    return f,ax

def inspect_multivariate_prediction(model, X, y, config):
    columns = get_labels()
    telescope = config["telescope"]
    
    # Make prediction
    pred = model.predict(X)

    # 
    idx=np.random.randint(0,X.shape[0])
        
    f, ax = plt.subplots(7, 1, sharex=True, figsize=(14,17))
    for i, col in enumerate(columns):
        ax[i].plot(np.arange(len(X[idx,:,i])), X[idx,:,i],label="Past")
        ax[i].plot(np.arange(len(X[idx,:,i]), len(X[idx,:,i])+telescope), y[idx,:,i], 
                    '.-', color='orange', markersize=12, label="Future")
        ax[i].plot(np.arange(len(X[idx,:,i]), len(X[idx,:,i])+telescope), pred[idx,:,i],
                    '.-', color='green', markersize=12, label="Prediction")
        ax[i].set_title(col)
        ax[i].legend()
    return f,ax

if __name__ == '__main__':
    print("Hey")



