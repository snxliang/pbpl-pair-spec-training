#data.py loads and processes the data into the following variables
#train_hits
#train_spectra
#test_hits
#test_spectra

# Importing relevant libraries
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

from constants import num_shots, num_of_training_cases, detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor, code_directory_path

# Loading the data
f = h5py.File(code_directory_path + "R.h5","r")
data = f["hits"]    # shape (64, 128, 1)

# Constructing the data structures to pass to the model
R = np.empty(shape = (64, detector_bin_count), dtype = object) #R matrix
train_hits = np.zeros(shape = (64, detector_bin_count, num_shots), dtype = object) #MODEL INPUT
train_spectra = np.zeros(shape = (64, 64), dtype = object) #MODEL OUTPUT

# Creating the arrays to contain the testing data. 
test_hits = np.zeros(shape = (1, detector_bin_count, num_shots), dtype = object)
test_spectra = np.zeros(shape = (1, 64), dtype = object)

energy_bins = f["num_events"][:]
noise_level = 1e04

indices_64 = np.linspace(0, 64, num_shots+1)
for i in range(0, len(indices_64)):
    indices_64[i] = np.round(indices_64[i])

indices_128 = np.linspace(0, 128, num_shots+1)
for i in range(0, len(indices_128)):
    indices_128[i] = np.round(indices_128[i])

# Loading in values for train_spectra and R matrix
j = 0
for i in range(0, 64):
	train_spectra[i][i] = energy_bins[i]
	vector = []
	for k in range(0, detector_bin_count):
		vector.append(detection_factor*sum(data[i][k])/1e09)
	R[i] = vector
	if (i != 0 and i in indices_64):
		j += 1
	train_hits[i,:,j] = np.dot(train_spectra[i][i], R[i])

# The following loop randomly generates num_of_training_cases training cases by randomly creating photon distributions and calculating 
# the associated electron-positron hits to provide to the model as training data.

for i in range(0, num_of_training_cases):
	arb_gamma_dist = np.zeros(shape=(1,64))
	# Creating an array of bins to generate a random number of photons
	# Consider generating random weights within the space of all betatron functions
	# Should be a 2D search space
	# This can yield a library of neural networks trained on different physical cases
	# Eg: energy_bin = [0, 24, 36, 52, 14, 9, 61] when i%12 = 7
	# energy_bin = np.random.randint(0, 64, 64)

	for k in range(0, 64):
		arb_gamma_dist[0][k] = np.random.randint(1e05, 1e10) #Adding a random number of photons between 1e05 and 1e10 in each bin based on energy_bin

	# Calculating the simulated hits in the PEDRO detector by multiplying the arbitrary gamma distrbution with the R matrix
	calculated_hits = np.zeros(shape=(1,detector_bin_count))
	calculated_hits[0] = np.dot(arb_gamma_dist[0], R)  
	calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data

	#FIX THIS
	j = 0
	vector = np.zeros((1, 128, num_shots))
	for i in range(0, 128):
		if (i in indices_128 and i != 0):
			j += 1
		vector[0,i,j] = calculated_hits[0][i]
	train_hits = np.concatenate((train_hits, vector))

	# Adding the simulated hits and gamma distribution to the training data
	print("train_hits shape: ", train_hits.shape)
	train_spectra = np.concatenate((train_spectra, arb_gamma_dist))

# The following loop is creating testing data for the model. 
# Consider adding noise to the testing data, but not to the training data?
test_cases = int(num_of_training_cases/10)
for i in range(0, test_cases):
	arb_gamma_dist = np.zeros(shape=(1,64))
	
	for k in range(0, 64):
		upper_lim = np.random.randint(5, 10)
		arb_gamma_dist[0][k] = np.random.randint(1e05, 10e10)
	
	calculated_hits = np.zeros(shape=(1,detector_bin_count))
	calculated_hits[0] = np.dot(arb_gamma_dist[0], R)
	#calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
	
	j = 0
	vector = np.zeros((1, 128, num_shots))
	for i in range(0, 128):
		if (i in indices_128 and i != 0):
			j += 1
		vector[0,i,j] = calculated_hits[0][i]
	test_hits  = np.concatenate((test_hits, vector))
	test_spectra = np.concatenate((test_spectra, arb_gamma_dist))

#train_hits = [train_hits]
test_hits = [test_hits]
train_hits_temp = train_hits
test_hits_temp = test_hits
test_hits = np.swapaxes(test_hits, 0, 1)
test_hits = np.swapaxes(test_hits, 1, 2)
#print(train_hits.shape, train_spectra.shape, test_hits.shape, test_spectra.shape)

# Converting the arrays into formats the model can process
train_hits = tf.convert_to_tensor(train_hits, dtype=float)
train_spectra = tf.convert_to_tensor(train_spectra, dtype=float)
test_hits = tf.convert_to_tensor(test_hits, dtype=float)
test_spectra = tf.convert_to_tensor(test_spectra, dtype=float)
