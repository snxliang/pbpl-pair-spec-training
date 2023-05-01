# Importing relevant libraries
import tensorflow as tf
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plot

from functions import gamma_dist_pred_and_plot, load_response, photon_spectral_density_func, recover_energy_dist, iterate_shepp_vardi, plot_spectrum
from constants import detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor, code_directory_path, num_of_training_cases
from data import train_hits, train_spectra, test_hits, test_spectra, R, energy_bins
from model import model

# To plot graphs, you need to use photon density against the energy. Divide the photons in each bin by (energy_max - energy_min) as the y-values. Plot using the step graph method with matplotlib (google it). 
# Currently, you're trying to find out how to calculated the energy bin lengths (the differences) and are having trouble keeping track of the units

# things to do
# generalize to n shots
# test loss and acc

# Use model to predict and plot from random data
e_bins = np.indices((1,64))
gamma_dist_pred_and_plot(e_bins[1][0])

plot.show()
quit()

# Regarding the last code block, if you see an indexing error involving the variable arb_gamma_distribution, rerun the block 1-2 more times and the error should be resolved. This is the result of the order in which the variables are loaded.

# For the .h5 file, use the following key:
# nlcs.h5 for Nonlinear Compton Scatter
# filamentation.h5 for Filamentation
# qed.h5 for Quantum Electrodynamics
with h5py.File(code_directory_path + 'qed.h5', 'r') as fin:
	y_experiment = fin['hits'][0,:]
	y_experiment = y_experiment.sum(axis=1)
	y_experiment = y_experiment.astype('float64')
	num_events_real = fin['num_events'][:]

with h5py.File(code_directory_path + "R.h5","r") as f:
	R_test, gamma_bins, detector_bins = load_response(f)

dE = gamma_bins[1:] - gamma_bins[:-1]
energy = gamma_bins[:-1] + 0.5*dE
x0 = photon_spectral_density_func(energy)

# The following set of lines (until y_experiment = np.dot(arb_gamma_distrbution[0], R)) can be used to implement different cases
# Uncomment the following line and all lines associated with each case to test it. Don't forget to comment the last 4 lines of 
# this code block to avoid loading the experimental cases

arb_gamma_distrbution = np.zeros(shape=(1,64)) #Setting the gamma distribution equal to 0

# Monoenergetic Spectrum
#arb_gamma_distrbution[0][2] = 1e08 #Adding one arbitrary spike (the second index indicates the bin)

noise_level = 1e02

# Bienergetic Spectrum
arb_gamma_distrbution[0][2] = 1e08 #Adding one arbitrary spike (the second index indicates the bin)
#arb_gamma_distrbutio[0][42] = 1e10  #Adding another arbitrary spike (the second index indicates the bin)

# Random Spectrum across all 64 bins
#for i in range(0, 64):
#	rand_freq = np.random.randint(0, 1e10)
#	arb_gamma_distrbution[0][i] = rand_freq

# Calculating the hits in each detector by multiplying the gamma distribution by the R matrix
#y_experiment = np.dot(arb_gamma_distrbution[0], R) + np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
y_experiment += np.random.poisson(noise_level, detector_bin_count) #Adding some noise to the data
gamma_dist = GeV * arb_gamma_distrbution[0]/dE
#Adding Gaussian noise
#Take peak spectrometer signal, /2^8, treat that a standard deviation for a GD with a mean of 0, add to spectrometer signal and make sure there are no negative values

# Reconstructing the spectrum using QR decomposition
qr_guess = recover_energy_dist(R, y_experiment)

# Converting the experimental values into a reconstructed spectrum using the ML model
output = model(tf.convert_to_tensor([y_experiment]))
num_events = output[:]

model_guess = num_events[0] 

# Reconstructing the spectrum using Maximum-Likelihood Estimation with the ML model's guess provided as the initial guess
mle_guess = model_guess
for i in range(5):
	mle_guess = iterate_shepp_vardi(mle_guess, R, y_experiment)

photon_spectral_density = model_guess[0]/dE
E_lim = [energy_bins[0], energy_bins[-1]]

# Code to write the information into a spectrum h5 file
#with h5py.File(code_directory_path + 'spectrum.h5', 'w') as fout:
#	fout['num_events'] = num_events_real
#	fout['energy_bins'] = gamma_bins/MeV
#	fout['photon_spectral_density'] = photon_spectral_density/(1/MeV)

matplotlib.rc('figure.subplot', right=0.97, top=0.96, bottom=0.15, left=0.13)
fig = plot.figure(figsize=(12.0, 7.0))
ax = fig.add_subplot(1, 1, 1)

with h5py.File(code_directory_path + 'spectrum.h5', 'r') as fin:
	energy_bins = fin['energy_bins'][:]*MeV
	photon_spectral_density = fin['photon_spectral_density'][:]*(1/MeV)
	num_events = fin['num_events'][:]

plot.xlabel('Gamma energy (GeV)', labelpad=0.0)
plot.ylabel(r'Photon density (1/Gev)', labelpad=0.0)
photon_spectral_density = num_events[0]/dE

#Insert variable into the plot_spectral_photon_density depending on whether you're running filamentation or qed respectively
filamentation_group = '/Filamentation/solid'
sfqed_group = '/SFQED/MPIK/LCS+LCFA_w2.4_xi7.2'

# If testing the arbitrary gamma distributions, comment out all of the following lines
#with h5py.File(code_directory_path + 'd2W.h5', 'r') as fin:
	#If running Nonlinear Compton scattering case, comment thef following line and uncomment the last line
	#gamma_dist = plot_spectral_photon_density(ax, fin, sfqed_group, E_lim, num_events, gamma_bins)

# Uncomment only when running Nonlinear Compton scattering case; else comment out
#gamma_dist = plot_spectral_photon_density_ncs(ax, code_directory_path +'nonlinear-compton-scattering.dat', E_lim, num_events, gamma_bins)

plot_spectrum(np.abs(model_guess.numpy()), dE, gamma_bins, "model", ax)
plot_spectrum(np.abs(mle_guess), dE, gamma_bins, "mle", ax)
plot_spectrum(np.abs(qr_guess[0]), dE, gamma_bins, "qr", ax)

# Setting the title and finalizing the visual appearance of the Reconstruction Plot
ax.set_title("Filamentation Reconstructed (using _____ approach) and Gamma Spectra")
ax.legend()
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
ax.yaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.yaxis.set_minor_locator(locmin)

ax.set_xlim(gamma_bins[0]/GeV, gamma_bins[-1]/GeV)
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12)
ax.xaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.xaxis.set_minor_locator(locmin)

plot.show()
quit()

#Junkyard
#Plotting the different reconstructions and the original gamma distribution for comparison
def temp_plot(guess, true_dist, title):
	plot.plot(np.abs(guess), label = title)
	plot.plot(true_dist*dE/GeV, label = "Gamma distribution")
	plot.xlabel("Energy Bin")
	plot.ylabel("Number of Photons")
	plot.yscale("log")
	plot.ylim([1e04, 2e08])
	plot.legend()

plot.plot(np.abs(qr_guess[0]), label = "QR based prediction")
plot.plot(np.abs(mle_guess), label = "MLE based prediction")
plot.plot(np.abs(model_guess), label = "Model based prediction")
plot.plot(gamma_dist*dE/GeV, label = "Gamma distribution")
plot.xlabel("Energy Bin")
plot.ylabel("Number of Photons")
plot.yscale("log")
plot.ylim([1e05, 2e08])
plot.legend()

temp_plot(qr_guess[0], gamma_dist, "QR based prediction")

#Plotting the different reconstructions and the original gamma distribution for comparison
temp_plot(mle_guess, gamma_dist, "MLE based prediction")

temp_plot(model_guess, gamma_dist, "Model based prediction")

gamma_bins[0:29]/GeV

plot.show()
