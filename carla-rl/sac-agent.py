import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque

from keras import Model, Input
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten, MaxPool2D, \
    MaxPooling2D, Dropout, concatenate, Lambda
from keras.optimizers import Adam, RMSprop

IM_HEIGHT = 128
IM_WIDTH = 128


# Warstwa krytyka - przyjmuje stan i akcję, zwraca wartość Q
def build_critic(state_shape, action_dim):
    state_input = Input(shape=state_shape)
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(state_input)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    action_input = Input(shape=(action_dim,))
    x = concatenate()([x, action_input])
    x = Dense(256, activation='relu')(x)
    q_output = Dense(1)(x)
    critic = Model(inputs=[state_input, action_input], outputs=q_output)
    critic.compile(optimizer=Adam(lr=0.0003), loss='mse')
    return critic

# Warstwa aktora - przyjmuje stan i zwraca rozkład prawdopodobieństwa akcji
def build_actor(state_shape, action_dim, max_action):
    state_input = Input(shape=state_shape)
    x = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(state_input)
    x = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(x)
    x = Flatten()(x)
    mean_output = Dense(256, activation='relu')(x)
    mean_output = Dense(action_dim, activation='tanh')(mean_output)
    mean_output = Lambda(lambda x: x * max_action)(mean_output)
    log_std_output = Dense(256, activation='relu')(x)
    log_std_output = Dense(action_dim)(log_std_output)
    actor = Model(inputs=state_input, outputs=[mean_output, log_std_output])
    return actor

# Funkcja losowości akcji
def sample_action(mu, log_std):
    std = tf.exp(log_std)
    eps = tf.random.normal(tf.shape(mu))
    return mu + eps * std

# Funkcja wykorzystywana do obliczania straty entropii
def entropy_loss(log_std):
    return 0.5 * tf.reduce_sum(log_std + np.log(2 * np.pi * np.e), axis=-1)

# Implementacja algorytmu SAC
class SAC:
    def __init__(self, state_shape, action_dim, max_action):
        self.critic1 = build_critic(state_shape, action_dim)
        self.critic2 = build_critic(state_shape, action_dim)
        self.actor = build_actor(state_shape, action_dim, max_action)
        self.target_entropy = -action_dim
        self.log_alpha = tf