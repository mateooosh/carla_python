import gym
import random
import numpy as np
from collections import deque

from keras import Model, Input
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten, MaxPool2D, \
    MaxPooling2D, Dropout, concatenate
from keras.optimizers import Adam, RMSprop

IM_HEIGHT = 128
IM_WIDTH = 128


class DQNAgent:
    def __init__(self, action_size, neural_network):
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        # self.epsilon = 1 if keep_learing else 0.05
        self.epsilon = 1.0
        self.epsilon_decay = 0.9985
        # self.epsilon_min = 0.01
        self.epsilon_min = 0.1
        self.learning_rate = 0.001
        self.model = self._build_model(neural_network)

    def _build_model(self, neural_network):
        if neural_network == 'model_3':
            input_image = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

            conv1 = Conv2D(16, (8, 8), activation='relu', padding='same')(input_image)
            p1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv1)
            conv2 = Conv2D(32, (4, 4), activation='relu')(p1)
            p2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv2)
            d1 = Dropout(0.3)(p2)
            flatten = Flatten()(d1)

            # drugie wejście dla wektorów
            input_vector = Input(shape=(4,))

            # warstwy gęste dla wektorów
            dense1 = Dense(128, activation='relu')(input_vector)

            # łączenie dwóch ścieżek
            concat = concatenate([flatten, dense1])

            # wyjście
            dense2 = Dense(256, activation='relu')(concat)
            d2 = Dropout(0.2)(dense2)
            output = Dense(self.action_size, activation='linear')(d2)

            model = Model(inputs=[input_image, input_vector], outputs=output)
            model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
            model.summary()
            return model

        else:
            return None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma + np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print(act_values[0], np.argmax(act_values[0]))
        return np.argmax(act_values[0])

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)
        self.epsilon = self.epsilon_min
