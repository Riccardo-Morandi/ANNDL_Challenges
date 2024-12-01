import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers

def build_simple_model(input_shape):
  Dropout = 0.1
  num_classes = 14

  # Build the neural network layer by layer
  input_layer = tfkl.Input(shape=input_shape, name='Input')


  x = tfkl.Conv2D(16, 3, padding='same', activation='relu')(input_layer)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.Conv2D(16, 3, padding='same', activation='relu')(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.MaxPooling2D()(x)
  x = tfkl.Conv2D(32, 3, padding='same', activation='relu')(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.Conv2D(32, 3, padding='same', activation='relu')(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.MaxPooling2D()(x)
  x = tfkl.Conv2D(64, 3, padding='same', activation='relu')(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.Conv2D(128, 3, padding='same', activation='relu')(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.MaxPooling2D()(x)
  x = tfkl.GlobalAveragePooling2D()(x)
  x = tfkl.Dropout(Dropout)(x)
  x = tfkl.Dense(128, activation='relu')(x)
  x = tfkl.Dense(num_classes,activation='softmax')(x)

  output_layer = x

  # Connect input and output through the Model class
  model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

  # Compile the model
  model.compile(loss=tfk.losses.CategoricalCrossentropy(), 
                optimizer=tfk.optimizers.Adam(learning_rate=1e-5), metrics='accuracy')

  # Return the model
  return model