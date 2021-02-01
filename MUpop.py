import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from MUsim import MUsim

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(mu, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

mu = MUsim()
mu.num_units = 32     # default number of units to simulate
mu.num_trials = 100    # default number of trials to simulate
mu.num_bins_per_trial = 1000 # amount of time per trial is (num_bins_per_trial/sample_rate)
mu.sample_rate = 1/(0.006)
_, latents = mu.sample_MUs(MUmode="lorenz")
sess1 = mu.simulate_session()
smth_sess1 = mu.convolve(sigma=30,target="session")
mu.see('rates',session=0)

latent_dim = 8

encoder_inputs = keras.Input(shape=(mu.num_bins_per_trial, mu.num_units))
x = layers.RNN(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.RNN(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

