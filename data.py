#data.py loads and processes the data into the following variables
#train_hits
#train_spectra
#test_hits
#test_spectra

# Importing relevant libraries
import tensorflow as tf
import numpy as np
import h5py
import csv

from constants import num_shots, num_of_training_cases, detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor, code_directory_path

# Toggle this to use saved data instead of creating new data
toggle = 0

if toggle:
    with open(code_directory_path + "train_hits.csv") as train_hits:
        train_hits = tf.convert_to_tensor(train_hits, dtype=float)
        train_spectra = tf.convert_to_tensor(train_spectra, dtype=float)
        test_hits = tf.convert_to_tensor(test_hits, dtype=float)
        test_spectra = tf.convert_to_tensor(test_spectra, dtype=float)
        quit()

# Loading the data
f = h5py.File(code_directory_path + "R.h5","r")
data = f["hits"]    # shape (64, 128, 1)

# Constructing the data structures to pass to the model
R = np.empty(shape = (64, detector_bin_count), dtype = object) #R matrix
train_hits = np.empty(shape = (64, detector_bin_count), dtype = object) #MODEL INPUT
train_spectra = np.zeros(shape = (64, 64), dtype = object) #MODEL OUTPUT

# Creating the arrays to contain the testing data. 
test_hits = np.zeros(shape = (1, detector_bin_count), dtype = object)
test_spectra = np.zeros(shape = (1, 64), dtype = object)

energy_bins = f["num_events"][:]
noise_level = 1e04

# Loading in values for train_spectra and R matrix
for i in range(0, 64):
	train_spectra[i][i] = energy_bins[i]
	vector = []
	for k in range(0, detector_bin_count):
		vector.append(detection_factor*sum(data[i][k])/1e09)
	R[i] = vector
	train_hits[i] = np.dot(train_spectra[i][i], R[i])

# The following loop randomly generates num_of_training_cases training cases by randomly creating photon distributions and calculating 
# the associated electron-positron hits to provide to the model as training data.
for i in range(0, num_of_training_cases):
	arb_gamma_distrbution = np.zeros(shape=(1,64))
	# Creating an array of bins to generate a random number of photons
	# Consider generating random weights within the space of all betatron functions
	# Should be a 2D search space
	# This can yield a library of neural networks trained on different physical cases
	# Eg: energy_bin = [0, 24, 36, 52, 14, 9, 61] when i%12 = 7
	# energy_bin = np.random.randint(0, 64, 64)

	for k in range(0, 64):
		arb_gamma_distrbution[0][k] = np.random.randint(1e05, 1e10) #Adding a random number of photons between 1e05 and 1e10 in each bin based on energy_bin

	# Calculating the simulated hits in the PEDRO detector by multiplying the arbitrary gamma distrbution with the R matrix
	calculated_hits = np.zeros(shape=(1,detector_bin_count))
	calculated_hits[0] = np.dot(arb_gamma_distrbution[0], R)  
	calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data

	# Adding the simulated hits and gamma distribution to the training data
	train_hits  = np.concatenate((train_hits, calculated_hits))
	train_spectra = np.concatenate((train_spectra, arb_gamma_distrbution))

# The following loop is creating testing data for the model. 
# Consider adding noise to the testing data, but not to the training data?
test_cases = int(num_of_training_cases/10)
for i in range(0, test_cases):
	arb_gamma_distrbution = np.zeros(shape=(1,64))
	
	for k in range(0, 64):
		upper_lim = np.random.randint(5, 10)
		arb_gamma_distrbution[0][k] = np.random.randint(1e05, 10e10)
	
	calculated_hits = np.zeros(shape=(1,detector_bin_count))
	calculated_hits[0] = np.dot(arb_gamma_distrbution[0], R)
	calculated_hits = calculated_hits + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
	
	test_hits  = np.concatenate((test_hits, calculated_hits))
	test_spectra = np.concatenate((test_spectra, arb_gamma_distrbution))

# Adapting data for num_shots shots
train_hits_64 = train_hits
train_spectra_64 = train_spectra
test_hits_64 = test_hits
test_spectra_64 = test_spectra
for i in range(1, num_shots):
    train_hits = np.concatenate((train_hits, train_hits_64), axis=1)
    train_spectra = np.concatenate((train_spectra, train_spectra_64), axis=1)
    test_spectra = np.concatenate((test_spectra, test_spectra_64), axis=1)
    test_hits = np.concatenate((test_hits, test_hits_64), axis=1)

# Converting the arrays into formats the model can process
train_hits = tf.convert_to_tensor(train_hits, dtype=float)
train_spectra = tf.convert_to_tensor(train_spectra, dtype=float)
test_hits = tf.convert_to_tensor(test_hits, dtype=float)
test_spectra = tf.convert_to_tensor(test_spectra, dtype=float)
