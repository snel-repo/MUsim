import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from MUsim import MUsim

mu1 = MUsim()
# GET LORENZ SIMULATED MOTOR UNITS
mu1.num_units = 30
mu1.sample_rate = 1/(0.006) # 166.7 Hz
mu1.MUthresholds_dist = 'uniform'
lorenz_units = mu1.sample_MUs(MUmode="lorenz")
# %% SIMULATE MOTOR UNITS SPIKING RULED BY LORENZ DYNAMICS
spikes = mu1.simulate_session()
mu1.see('spikes') # plot spike response
# %% CONVOLVE AND PLOT SMOOTHED SPIKES
smooth = mu1.convolve(target='session')
mu1.see('rates') # plot smoothed spike response
# %% VIEW LORENZ ATTRACTOR
mu1.see('lorenz')