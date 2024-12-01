import tensorflow as tf
from keras import backend as K
tfk = tf.keras
tfkl = tf.keras.layers

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def build_CONV_LSTM_model(input_shape, output_shape):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    convlstm = tfkl.Bidirectional(tfkl.LSTM(64, return_sequences=True))(input_layer)
    convlstm = tfkl.Conv1D(128, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.MaxPool1D()(convlstm)
    convlstm = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(convlstm)
    convlstm = tfkl.Conv1D(256, 3, padding='same', activation='relu')(convlstm)
    convlstm = tfkl.GlobalAveragePooling1D()(convlstm)
    convlstm = tfkl.Dropout(.5)(convlstm)

    # In order to predict the next values for more than one channel,
    # we can use a Dense layer with a number given by telescope*num_channels,
    # followed by a Reshape layer to obtain a tensor of dimension 
    # [None, telescope, num_channels]
    dense = tfkl.Dense(output_shape[-1]*output_shape[-2], activation='relu')(convlstm)
    output_layer = tfkl.Reshape((output_shape[-2],output_shape[-1]))(dense)
    output_layer = tfkl.Conv1D(output_shape[-1], 1, padding='same')(output_layer)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.MeanSquaredError(), optimizer=tfk.optimizers.Adam(), metrics=['mse'])
    # model.compile(optimizer = "rmsprop", loss = root_mean_squared_error, metrics =["mse"])
    
    # Return the model
    return model

def build_model2(input_shape, output_shape):
    drp = 0.2
    out_size = tf.math.reduce_prod(output_shape).numpy()
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    x = input_layer
    
    x = tfkl.Bidirectional(tfkl.LSTM(128, return_sequences=True))(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPool1D()(x)    
    
    x = tfkl.GlobalAveragePooling1D()(x)
    x = tfkl.Dropout(.5)(x)

    # Dense
    x = tfkl.Dense(256, activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    
    x = tfkl.Dense(out_size, activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Reshape(output_shape)(x)
    x = tfkl.Conv1D(output_shape[1], 1, padding='same')(x)

    output_layer = x

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.MeanSquaredError(), 
                  optimizer=tfk.optimizers.Adam(learning_rate=5e-4), metrics=['mse'])
    return model

def build_model3(input_shape, output_shape):
    drp = 0.2
    out_size = tf.math.reduce_prod(output_shape).numpy()
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')
    x = input_layer
    
    x = tfkl.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.LSTM(64, return_sequences=True)(x)
    x = tfkl.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.MaxPool1D()(x)
    
    x = tfkl.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.LSTM(64, return_sequences=True)(x)
    x = tfkl.Dropout(drp)(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tfkl.Dense(64, activation='relu')(x)
    x = tfkl.Dropout(drp)(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(out_size, activation=None)(x)
    x = tfkl.Reshape(output_shape)(x)


    output_layer = x

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    optimizer = tfk.optimizers.Adam(learning_rate=1e-3)
    loss=tf.keras.losses.Huber() # tfk.losses.MeanSquaredError()
    metrics=["mae","mse"]
    model.compile(loss=loss,optimizer=optimizer, metrics=metrics)
    return model

def define_models(n_input, n_output, n_units):
	# define training encoder
	encoder_inputs = tfkl.Input(shape=(None, n_input))
	encoder = tfkl.LSTM(n_units, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	encoder_states = [state_h, state_c]
	# define training decoder
	decoder_inputs = tfkl.Input(shape=(None, n_output))
	decoder_lstm = tfkl.LSTM(n_units, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = tfkl.Dense(n_output, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = tfk.Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
	encoder_model = tfk.Model(encoder_inputs, encoder_states)
	# define inference decoder
	decoder_state_input_h = tfkl.Input(shape=(n_units,))
	decoder_state_input_c = tfkl.Input(shape=(n_units,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = tfk.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
	return model, encoder_model, decoder_model

def build_encoder_decoder(input_shape,output_shape, 
                          latent_dim = 64,
                          dropout = 0.2):
    # out_size = tf.math.reduce_prod(output_shape).numpy()

    # First branch of the net is an lstm which finds an embedding for the past
    past_inputs = tf.keras.Input(shape=input_shape, name='past_inputs')
    
    # future_inputs = past_inputs[:,-output_shape[0]:,:output_shape[1]]
    future_inputs = past_inputs[:,:,:output_shape[1]]
    
    # Encoding the past
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(past_inputs)
    
    # future_inputs = tf.keras.Input(shape=output_shape, name='future_inputs')
    
    # Combining future inputs with recurrent branch output
    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    x = decoder_lstm(future_inputs,initial_state=[state_h, state_c])
    
    # Not sure if this is the best way
    # To reduce the number of channels to something specific
    x = tfkl.MaxPool1D()(x)
    x = tfkl.Conv1D(latent_dim, input_shape[0]//2-output_shape[0]+1, activation='relu')(x)
    
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(16, activation='relu')(x)
    # x = tfkl.Dropout(dropout)(x)
    # x = tfkl.BatchNormalization()(x)
    # x = tfkl.Dense(out_size, activation=None)(x)
    # x = tfkl.Reshape(output_shape)(x)
    # output = x
    
    x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    x = tfkl.Dropout(dropout)(x)
    x = tfkl.BatchNormalization()(x)
    x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    x = tfkl.Dropout(dropout)(x)
    x = tfkl.BatchNormalization()(x)
    # x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    # x = tfkl.Dropout(dropout)(x)
    # x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(output_shape[1], activation=None)(x)
    output = x
    
    # model = tfk.models.Model(inputs=[past_inputs, future_inputs], outputs=output)
    model = tfk.models.Model(inputs = past_inputs, outputs=output)
    
    
    optimizer = tfk.optimizers.Adam()
    loss = tfk.losses.Huber()
    model.compile(loss=loss, optimizer=optimizer, metrics=["mae"])
    return model

if __name__ == '__main__':
    window = 200
    telescope = 96
    
    input_shape = (window,9)
    output_shape = (telescope,7)
    
    # # model = build_CONV_LSTM_model(input_shape, output_shape)
    # model = build_model3(input_shape, output_shape)
    # model.summary()
    
    # n_input = 100
    # n_output = 3
    # n_units = 128
    # model, encoder_model, decoder_model = define_models(n_input, n_output, n_units)
    
    
    model = build_encoder_decoder(input_shape,output_shape, latent_dim = 64)
    model.summary()
    
    # tf.keras.utils.plot_model(model, show_shapes=True)
    
    # a = tf.zeros((2,200,9))
    # out = model.predict(a)
    
    