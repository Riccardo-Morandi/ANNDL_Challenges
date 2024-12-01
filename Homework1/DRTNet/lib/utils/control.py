import os
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

def set_seed(seed=12):
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return None

def get_labels():
    labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']
    return labels

def combine_dicts(dict1,dict2):
    """ Used to append fine-tune history to training history
    Assumes that the dicts shares keys
    """
    # dict3 = np.copy(dict1)
    for key in dict1.keys():
        dict1[key] += dict2[key]
    return dict1

def visualize_data(loader):
    labels = get_labels()
    X,y = next(loader)
    
    MAX_N = 5
    N = X.shape[0]//2 if X.shape[0]//2 < MAX_N else MAX_N
    f,ax = plt.subplots(2,N, figsize=(14,6))
    ax = ax.flatten()
    for i in range(2*N):
      if np.max(X) > 1.0:
        ax[i].imshow(np.uint8(X[i]))
      else:
        ax[i].imshow(X[i])
      ax[i].set_title(labels[np.argmax(y[i])])
    f.suptitle('Examples of training set with label')
    plt.show()
    print("Data range: [", np.min(X), ",", np.max(X),"]")
    return f,ax


def visualize_predictions(model, loader):
    labels = get_labels()
    X,y = next(loader)
    
    # Get prediction
    y_pred = model.predict(X)
    
    if y_pred.ndim==2:
        y_pred = np.argmax(y_pred,axis=-1)
    
    MAX_N = 5
    N = X.shape[0]//2 if X.shape[0]//2 < MAX_N else MAX_N
    f,ax = plt.subplots(2,N, figsize=(14,6))
    ax = ax.flatten()
    for i in range(2*N):
      if np.max(X) > 1.0:
        ax[i].imshow(np.uint8(X[i]))
      else:
        ax[i].imshow(X[i])
      ax[i].set_title(labels[np.argmax(y[i])]+"/"+labels[y_pred[i]])
    f.suptitle('Correct/Prediction')
    plt.show()
    return f,ax

def compute_confusion_matrix(model, loader):
    cm = np.zeros((14,14))
    for i in tqdm(range(len(loader)),total=len(loader),
                  desc='Running '):
        X,y = next(loader)
    
        y_hat = model.predict(X)
        if y_hat.ndim == 2:
            y_hat_max = np.argmax(y_hat, axis=-1)
        else:
            y_hat_max = y_hat

        confusion = confusion_matrix(np.argmax(y, axis=-1),y_hat_max,labels=list(range(14)))
        cm += confusion
    return cm

def plot_confusion_matrix(cm, normalize = True, show = True):
    
    if normalize:
        cmm = cm/np.sum(cm,axis=0)
    else:
        cmm = cm

    f,ax = plt.subplots(figsize=(10,3))
    sns.heatmap(cmm.T, xticklabels=get_labels(), yticklabels=get_labels())
    ax.set_xlabel('True labels')
    ax.set_ylabel('Predicted labels')
    if normalize:
        ax.set_title('Normalized confusion matrix')
    else:
        ax.set_title('Confusion matrix')
    if show:
        plt.show()
    return f,ax


def CM_whole_dataset(cm_train,cm_val):
    """ Shows a Confusion matrix for both training and validation sets
    """
    # Conput accuracy
    acc_train = np.sum(np.diag(cm_train))/np.sum(cm_train)
    acc_val = np.sum(np.diag(cm_val))/np.sum(cm_val)

    f,ax = plt.subplots(1,2,figsize=(14,5))
    sns.heatmap(cm_train.T, xticklabels=get_labels(), yticklabels=get_labels(),
                ax=ax[0])
    ax[0].set_xlabel('True labels')
    ax[0].set_ylabel('Predicted labels')
    ax[0].set_title("Training Set, accuracy: "+str(round(acc_train,4)))
    sns.heatmap(cm_val.T, xticklabels=get_labels(), yticklabels=get_labels(),
                ax=ax[1])
    ax[1].set_xlabel('True labels')
    ax[1].set_ylabel('Predicted labels')
    ax[1].set_title("Validation Set, accuracy: "+str(round(acc_val,4)))
    f.suptitle('Confusion matrices')       
    return f,ax




