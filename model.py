#Importing relevant libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from constants import num_shots, num_of_training_cases, detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor
from data import train_hits, train_spectra, test_hits, test_spectra

def create_model_1():
	model = keras.Sequential()
	model.add(keras.Input(shape=(128, num_shots)))
	#model.add(keras.Input(shape=(128*num_shots)))
	model.add(layers.Flatten())
	model.add(layers.Dense(64))
	model.add(layers.Dense(64))
	model.summary()

	#Learning rate of 0.005 was chosen after hyperparameter tuning
	opt = keras.optimizers.Adam(learning_rate=0.005)

	model.compile(optimizer=opt,loss=tf.keras.losses.mse, metrics='accuracy') 

	return model

def create_model_2():
    model = keras.models.Sequential()

    model.add(keras.layers.LSTM(64, input_shape=(128,num_shots), return_sequences=True, name='inputlstm1'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.LSTM(64, return_sequences=True,name='lstm2'))
    model.add(keras.layers.Dropout(0.2))

    # The last layer of Stacked LSTM need not to return the input sequences
    model.add(keras.layers.Dense(64, activation='relu', name='dense1'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(64, activation='softmax', name='denseoutput2'))


    # Compile model
    model.compile(
        loss='mse',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return model 

model = create_model_1()

print(train_hits.shape, train_spectra.shape)
model.fit(train_hits, train_spectra, epochs = 400, verbose=0)
print("Evaluate on test data")
#results = model.evaluate(test_hits, test_spectra)#print("test loss, test acc:", results)
