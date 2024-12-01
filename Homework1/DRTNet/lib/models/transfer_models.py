import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

def build_transfer_model(transfer_model, input_shape, 
                         preprocess_input, lr = 0.001):

    # Build the neural network layer by layer
    inputs = tfkl.Input(shape=input_shape, name='Input')

    # Freeze tranfer model parameters
    transfer_model.trainable = False

    # Preprocess
    x = tfkl.Rescaling(255., offset=0)(inputs) # Go from [0,1] to [0,255]
    x = preprocess_input(x)

    x = transfer_model(x, training=False)
    x = tfkl.GlobalAveragePooling2D()(x)
    x = tfkl.Dropout(0.2)(x)
    x = tfkl.Dense(256)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Activation('relu')(x)
    x = tfkl.Dropout(0.2)(x)
    x = tfkl.Dense(14,activation='softmax')(x)
    outputs = x

    # For ordering of BN, Dropout, LL and Activation see:
    # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout


    # Connect input and output through the Model class
    model = tf.keras.Model(inputs, outputs, name='Model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(learning_rate=lr), metrics='accuracy')
    return model

if __name__ == '__main__':
    from tensorflow.keras.applications.vgg19 import preprocess_input
    IMG_SHAPE = (224,224,3)
    transfer_model = tfk.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SHAPE,
        classifier_activation='softmax'
    )
    
    model = build_transfer_model(transfer_model,IMG_SHAPE, preprocess_input)
    model.summary()