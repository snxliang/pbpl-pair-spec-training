#Importing relevant libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from constants import num_shots, detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor
from data import train_hits, train_spectra, test_hits, test_spectra

#Creating the model
#model = keras.Sequential(
#  [ 
#   layers.Dense(64, activation='linear', use_bias=True, input_shape=(128*num_shots,)), 
#   layers.Dense(64*num_shots, activation='linear', use_bias=True)
#   ]
#)

model = keras.Sequential()
model.add(layers.Dense(64, input_shape=(128*num_shots,)))
model.add(layers.Dense(64*num_shots))

model.summary()

#Learning rate of 0.005 was chosen after hyperparameter tuning
opt = keras.optimizers.Adam(learning_rate=0.005)

model.compile(optimizer=opt,loss=tf.keras.losses.mse, metrics='accuracy') 
model.fit(train_hits, train_spectra, epochs = 400, verbose=0)
print("Evaluate on test data")
results = model.evaluate(test_hits, test_spectra)
print("test loss, test acc:", results)
