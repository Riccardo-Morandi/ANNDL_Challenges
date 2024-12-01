import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys

def normalize_img(img):
    if np.max(img)==np.min(img):
        return img
    else:
        normalized = (img-np.min(img))/(np.max(img)-np.min(img))
        return normalized
    return None

# def get_blur_layers(IMG_SIZE, 
#                         blur_levels = [3,5,11],
#                         thresholds = [0.2, 0.2, 0.4],
#                         factors = [1, 2, 5]):
        
#     combined = np.zeros(IMG_SIZE)
#     for bl,th,fc in zip(blur_levels,thresholds,factors):
#         layer = np.random.uniform(size=IMG_SIZE)
#         layer[layer <= 1-th] = 0
#         layer = normalize_img(cv2.blur(layer,(bl,bl)))
#         layer[layer <= 1-th] = 0
#         combined += layer*fc
#     normalized = normalize_img(combined)
#     return normalized


def custom_preprocess(image, 
                          probs = [0.3, 0.4, 0.05, 0.3, 0.3]):
                    
    """ probs are a list of length 5
      prob[0] constrols the noise added to the leaves
      prob[1] constrols the noise added to the black background
      prob[2] constrols swapping of color channels
      prob[3] constrols HSV hue of the image
      prob[4] constrols blurring of image
      """
    # Generate random values
    A,B,C,D,E = np.random.rand(5)
    
    # Define propabilites for each of the three augmentations
    if len(probs) != 5:
        raise ValueError("Lenght of threshold should be 5")
    else:
        thresholds = probs
    im_max = image.max()
    im_min = image.min()
        
    # Adds noise to the Leafs
    if A <= thresholds[0]:
        N = image[image != 0].shape
        values = np.random.uniform(low=im_min, high=im_max, 
                                   size=N)
        
        image[image != 0] += values*0.3
        image = normalize_img(image)*im_max
    
    # Adds noise in the black background
    if B <= thresholds[1]:
        BACKGROUND_VALUE = 0 # Here I assume that 0 is the background colour
        size = image[image==BACKGROUND_VALUE].shape[0] 
        values = np.random.uniform(low=image.min(), high=image.max(), size=(size,))
        image[image==0] = values
    
    # Swap color channels
    if C <= thresholds[2]:
        dims = np.arange(3)
        np.random.shuffle(dims)
        image = image[...,[dims[0],dims[1],dims[2]]]
    
    # Change the hue of the image
    if D <= thresholds[3]:
        image = np.uint8(np.array(image))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image = image.astype(np.float32)
        
    # Blurs the image slightly
    if E <= thresholds[4]:
        image = cv2.blur(image,(5,5))
    return image

def get_loaders(batch_size,IMG_SHAPE=(224,224,3),validation_split = 0.1,
                training_dir = 'dataset/training/',
                auglevel = 0,
                seed=33):
    """
    
    auglevel = 0,1,2   # Augmentation gets more intense the
    """
    # See following for how train/val split is enforced when from same directory
    # https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
    
    labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']
    if auglevel == 0:
        aug_train = ImageDataGenerator(validation_split=validation_split,
                                     rescale=1/255.
                                      )
    elif auglevel == 1:
        aug_train = ImageDataGenerator(validation_split=validation_split,
                                 rotation_range = 45,
                                 width_shift_range = 0.3,
                                 height_shift_range = 0.3,
                                 brightness_range = [0.3,1.4],
                                 shear_range = 10,
                                 zoom_range = [0.3,1.5],
                                 channel_shift_range = 40,
                                 horizontal_flip  = True,
                                 vertical_flip = True,
                                 rescale=1/255.
                                  )
    else:
        aug_train = ImageDataGenerator(validation_split=validation_split,
                                 rotation_range = 45,
                                 width_shift_range = 0.3,
                                 height_shift_range = 0.3,
                                 brightness_range = [0.3,1.4],
                                 shear_range = 10,
                                 zoom_range = [0.3,1.5],
                                 channel_shift_range = 40,
                                 horizontal_flip  = True,
                                 vertical_flip = True,
                                 preprocessing_function = custom_preprocess,
                                 rescale=1/255.
                                  )
    aug_val = ImageDataGenerator(validation_split=validation_split,
                                 rescale=1/255.
                                  )
    train_loader = aug_train.flow_from_directory(directory=training_dir,
                                                           target_size=IMG_SHAPE[:-1],
                                                           color_mode='rgb',
                                                           classes=labels, # can be set to labels
                                                           class_mode='categorical',
                                                           batch_size= batch_size,
                                                           shuffle=True,
                                                           seed=seed,
                                                           subset='training')
    val_loader = aug_val.flow_from_directory(directory=training_dir,
                                                           target_size=IMG_SHAPE[:-1],
                                                           color_mode = 'rgb',
                                                           classes=labels, # can be set to labels
                                                           class_mode='categorical',
                                                           batch_size= batch_size,
                                                           shuffle=True,
                                                           seed=seed,
                                                           subset='validation')
    return train_loader,val_loader

def get_submission_loader(batch_size,IMG_SHAPE=(224,224,3),validation_split = 0.1,
                training_dir = 'dataset/training/', seed=33):
    
    labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']
    aug_val_sub = ImageDataGenerator(validation_split=validation_split)

    val_loader_sub = aug_val_sub.flow_from_directory(directory=training_dir,
                                                           target_size=IMG_SHAPE[:-1],
                                                           color_mode = 'rgb',
                                                           classes=labels, # can be set to labels
                                                           class_mode='categorical',
                                                           batch_size= batch_size,
                                                           shuffle=True,
                                                           seed=seed,
                                                           subset='validation')
    return val_loader_sub

#%%
if __name__ == '__main__':
    sys.path.append("..")
    import utils as util
    training_dir = '../../dataset/balanced/'
    # Set seed
    seed = 42
    util.set_seed(seed) 
    
    # Set parameters
    batch_size = 16
    IMG_SHAPE=(224,224,3)
    validation_split = 0.1
    auglevel = 2

    train_loader,val_loader = get_loaders(batch_size,IMG_SHAPE,validation_split,
                    training_dir,auglevel,seed=seed)
    
    
    # Visualize data
    util.visualize_data(train_loader)
    
    X,y = next(train_loader)
    
    #%%
    # # Check balance of dataset
    # vals = np.zeros((14),dtype=int)
    
    # from tqdm import tqdm
    # for i in tqdm(range(len(val_loader)),total=len(val_loader),
    #           desc='Running '):
    #     X,y = next(val_loader)
    #     classes = np.argmax(y,axis=-1)
    #     for classs in classes:
    #         vals[classs] += 1
    # print(vals)

    
    
    
    
    
    
    
    
    
    