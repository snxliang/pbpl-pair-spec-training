import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plot

from constants import num_shots, detector_bin_count, twopi, MeV, GeV, detection_factor, model_correction_factor
from data import train_hits, train_spectra, test_hits, test_spectra, R
from model import model

#Declaring relevant functions
def iterate_shepp_vardi(x0, R, y):
	#x0 has shape num_shots*64 but R has shape 64x64 which needs to be fixed
	y0 = np.matmul(x0, R)
	mask = (y0 != 0)
	yrat = np.zeros_like(y)
	yrat[mask] = y[mask]/y0[mask]
	return (x0/R.sum(axis=1)) * np.matmul(R, yrat)

def load_response(fin):
	num_events = fin['num_events'][:].flatten()
	num_gamma_bins = len(num_events)
	gamma_bins = fin['i0'][:]*MeV
	hits = fin['hits'][:]
	detector_bins = fin['detector_bin'][:]
	photon_bins = fin['photon_bin'][:]*MeV

	# sum over photon energies
	hits = hits.sum(axis=2) * detection_factor

	hits = hits.astype('float64')
	num_cells = hits.shape[1]

	R = hits/num_events[:,np.newaxis]
	return R, gamma_bins, detector_bins

def photon_spectral_density_func(E):
	E0 = 1*GeV
	sigma0 = 2*GeV
	total_num_photons = 1e10
	A0 = total_num_photons/np.sqrt(twopi*sigma0**2)
	return A0 * np.exp(-(E-E0)**2/(2*sigma0**2))

#Plotting spectral photon density for the Nonlinear Compton Scattering case
def plot_spectral_photon_density_ncs(ax, filename, E_lim, num_photons_simulated, gamma_bins):
	photon_energy, spectral_photon_density = np.loadtxt(filename).T
	photon_energy *= GeV
	spectral_photon_density *= (1/(0.05*GeV))

	photon_energy, spectral_photon_density = resample(
		photon_energy, spectral_photon_density, 1000)

	mask = spectral_photon_density>0

	idx = range(*photon_energy.searchsorted(E_lim))
	num_photons_lim = simps(spectral_photon_density[idx], photon_energy[idx])
	alpha = num_photons_simulated/num_photons_lim

	ax.loglog(
		np.concatenate(((0.0,), photon_energy[mask]/GeV)),
		alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))),
		linewidth=0.6, label='Incident gamma spectrum')
	return generate_short_true_spectrum(np.concatenate(((0.0,), photon_energy[mask]/GeV)),
		alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))))

	#photon_energy[mask][1:]/GeV * (alpha*(spectral_photon_density[mask][1:] - spectral_photon_density[mask][:-1])/(1/GeV))
	
def resample(x, y, N):
	# import ipdb
	# ipdb.set_trace()
	x_y = interp1d(x, y, kind='linear')
	x0 = x[0]
	x1 = x[-1]
	u = 10**np.linspace(np.log10(x0), np.log10(x1), N)
	u[0] = x0
	u[-1] = x1
	v = x_y(u)
	return u, v

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array-value)).argmin()
  return idx

def plot_spectral_photon_density(
		ax, fin, group_name, E_lim, num_photons_simulated, gamma_bins):
	mrad = 0.001
	joule = 1.0
	g = fin[group_name]
	photon_energy = g['energy'][:]*MeV
	thetax = g['thetax'][:]*mrad
	thetay = g['thetay'][:]*mrad
	d2W = g['d2W'][:]*joule/(mrad**2*MeV)
	dthetax = thetax[1]-thetax[0]
	dthetay = thetay[1]-thetay[0]

	spectral_energy_density = d2W.sum(axis=(1,2))*dthetax*dthetay
	spectral_photon_density = spectral_energy_density/photon_energy

	mask = spectral_energy_density>0

	idx = range(*photon_energy.searchsorted(E_lim))
	num_photons_lim = simps(spectral_photon_density[idx], photon_energy[idx]) #out of bounds error on index???
	alpha = num_photons_simulated/num_photons_lim

	ax.loglog(
		np.concatenate(((0.0,), photon_energy[mask]/GeV)),
		alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))),
		linewidth=0.6, label='Incident gamma spectrum')
	return generate_short_true_spectrum(np.concatenate(((0.0,), photon_energy[mask]/GeV)),
		alpha*np.concatenate(((0.0,), spectral_photon_density[mask]/(1/GeV))))
	#photon_energy[mask]/GeV * (alpha * spectral_photon_density[mask]/(1/GeV))
	
def plot_spectrum(spec_to_plot, dE, gamma_bins, title, ax):
  spectral_density = spec_to_plot/dE
  ax.loglog(
		gamma_bins/GeV,
		np.concatenate(((0.0,),spectral_density/(1/GeV))),
		linewidth=0.6, label=title, ds = "steps")

def generate_short_true_spectrum(real_x_values, real_y_values):
	with h5py.File('/content/drive/MyDrive/Colab Notebooks/PBPL/spectrum.h5', 'r') as fin:
		energy_bins = fin['energy_bins'][:]*MeV
		photon_spectral_density = fin['photon_spectral_density'][:]*(1/MeV)
		num_events = fin['num_events'][:]

	x_values = energy_bins[0:-1]/GeV
	y_values = photon_spectral_density/(1/GeV)
	
	short_gamma_distribution = []

	for i in range(0, len(x_values)):
		index = find_nearest(real_x_values, x_values[i])
		short_gamma_distribution.append(real_y_values[index])
	return short_gamma_distribution

# This function will attempt to reconstruct a given (or random) gamma distribution after converting into a PEDRO spectrum
def gamma_dist_pred_and_plot(energy_bin, dist = np.zeros(shape=(1,64))):
	arb_gamma_distrbution = np.zeros(shape=(1,64))

	# if dist not specified, create random dist instead
	if (np.array_equal(dist,arb_gamma_distrbution)):
		for i in range(0, len(energy_bin)):
			#Decreasing Sigmoid: 1e11*(1 - 1/(1+np.exp(-((i+1)/8 - 4)))), 1e11*(1 - 1/(1+np.exp(-((i+0)/8 - 4))))
			#Exponential: 1e10*np.exp(i/64), 1e10*np.exp((i+1)/64)
			#Logarithmic: (64-(i+1))*1e10/64, (64-i)*1e10/64 
			#Random: 0, 1e10
			rand_freq = np.random.randint(0, 1e10)
			arb_gamma_distrbution[0][energy_bin[i]] = rand_freq
	else:
		arb_gamma_distrbution[0] = dist

	# Calculating the simulated spectrum by multiplying the distribution with the R matrix
	y_experiment = np.dot(arb_gamma_distrbution[0], R)

	# Adapting for multishot
	y_experiment_64 = y_experiment
	arb_gamma_distrbution_64 = arb_gamma_distrbution

	for i in range(1,num_shots):
		y_experiment = np.concatenate((y_experiment, y_experiment_64))
		arb_gamma_distrbution = np.concatenate((arb_gamma_distrbution, arb_gamma_distrbution_64), axis=1)

	# Applying the ML model to reconstruct the gamma distribution
	output = model(tf.convert_to_tensor([y_experiment]))
	num_events = output.numpy()[:]
	x = num_events[0]

	# Using the ML reconstruction as the initial guess for MLE, commented out since iterate_shepp_vardi not working with multishot yet
	#x_s = x
	#for i in range(75):
	#	x_s = iterate_shepp_vardi(x_s, R, y_experiment)

	scale_factor_list = []
	
	# Predicts the distribution 100 times and calculates the scale difference between the first element of the guess and original gamma distribution each time
	for i in range (0, 100):
		output=model(tf.convert_to_tensor([y_experiment]))
		num_events = output.numpy()[:]
		x=num_events[0]
		scale_factor_list.append(arb_gamma_distrbution[0][0]/x[0])
	
	# Calculating the average scale factor and printing it
	scale_factor = np.mean(scale_factor_list)
	scale_factor = 1 # why

	# Plotting the ML guess (with the scale_factor correction) with the original gamma distribution
	gamma_dist_plot(x*scale_factor, arb_gamma_distrbution[0])

def gamma_dist_plot(prediction, true):
	x = range(0, 64*num_shots)
	plot.figure().set_figwidth(6.4*num_shots)
	#qr_guess = recover_energy_dist(R, np.dot(prediction, R))
	plot.step(x, prediction, label = "Model based prediction")
	#plot.plot(qr_guess[0], label = "QR based prediction")
	plot.step(x, true, label = "Gamma distribution")
	plot.title("Comparison of Gamma Energy Distributions")
	plot.xlabel("Energy Bin")
	plot.ylabel("Number of Photons")
	plot.yscale("log")
	plot.legend()

#Implementation of the QR decomposition-based reconstruction method
def recover_energy_dist(R, spectrum): #R*guess = spectrum
	Q,S = np.linalg.qr(np.matrix.transpose(R), mode = "complete") #Q*S*guess = spectrum
	b = np.array([np.matrix.transpose(Q).dot(spectrum)], dtype="float") #S*guess = Q^t * spectrum = b
	guess = np.linalg.lstsq(S, b[0], rcond=None) # || S*guess - b ||_2 = 0
	return guess
