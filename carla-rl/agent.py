import gym
import random
import numpy as np
from collections import deque

from keras import Model
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Activation, AveragePooling2D, Flatten, MaxPool2D, \
    MaxPooling2D, Dropout
from keras.optimizers import Adam, RMSprop

IM_HEIGHT = 256
IM_WIDTH = 512


class DQNAgent:
    def __init__(self, action_size, neural_network, keep_learing, model_name):
        self.action_size = action_size
        self.keep_learing = keep_learing
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        # self.epsilon = 1 if keep_learing else 0.05
        self.epsilon = 1.0
        self.epsilon_decay = 0.998
        # self.epsilon_min = 0.01
        self.epsilon_min = 0.05
        self.learning_rate = 0.001
        self.model = self._build_model(neural_network)

        if keep_learing:
            self.load(model_name)

    def _build_model(self, neural_network):
        # model = Sequential()
        # model.add(Conv2D(64, (3, 3), input_shape=(IM_HEIGHT, IM_WIDTH, 3), padding='same', activation='relu'))
        # # model.add(Activation('relu'))
        # model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # # model.add(Activation('relu'))
        # model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        # # model.add(Activation('relu'))
        # model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))
        # model.add(Flatten())
        # model.add(Dense(self.action_size, activation="linear"))


        # model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(256, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(self.action_size, activation='softmax'))


        # LeNet-5
        if neural_network == 'LeNet':
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IM_HEIGHT, IM_WIDTH, 3)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.action_size, activation='linear'))
            model = Model(inputs=model.input, outputs=model.output)
            model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
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
