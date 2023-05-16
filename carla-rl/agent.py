import random
from collections import deque

import numpy as np
from keras import Model, Input
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout, concatenate, MaxPooling2D
from keras.optimizers import Adam

IM_HEIGHT = 128
IM_WIDTH = 128

class DQNAgent:
    def __init__(self, action_size, neural_network):
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9985
        self.epsilon_min = 0.15
        self.learning_rate = 0.001
        self.model = self._build_model(neural_network)

    def _build_model(self):
        input_image = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        conv1 = Conv2D(16, (8, 8), activation='relu', padding='same')(input_image)
        p1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv1)
        conv2 = Conv2D(32, (4, 4), activation='relu')(p1)
        p2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv2)
        d1 = Dropout(0.3)(p2)
        flatten = Flatten()(d1)

        input_vector = Input(shape=(4,))
        dense1 = Dense(128, activation='relu')(input_vector)
        concat = concatenate([flatten, dense1])
        dense2 = Dense(256, activation='relu')(concat)
        d2 = Dropout(0.2)(dense2)
        output = Dense(self.action_size, activation='linear')(d2)

        model = Model(inputs=[input_image, input_vector], outputs=output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model

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


class ActorCritic:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9985
        self.epsilon_min = 0.15
        self.learning_rate = 0.001
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        # Build actor model
        input_image = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        conv1 = Conv2D(16, (8, 8), activation='relu', padding='same')(input_image)
        p1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv1)
        conv2 = Conv2D(32, (4, 4), activation='relu')(p1)
        p2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv2)
        d1 = Dropout(0.3)(p2)
        flatten = Flatten()(d1)

        input_vector = Input(shape=(4,))
        dense1 = Dense(128, activation='relu')(input_vector)
        concat = concatenate([flatten, dense1])
        dense2 = Dense(256, activation='relu')(concat)
        d2 = Dropout(0.2)(dense2)
        actor_output = Dense(self.action_size, activation='linear')(d2)
        model = Model(inputs=[input_image, input_vector], outputs=actor_output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model

    def build_critic(self):
        # Build critic model
        input_image = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
        conv1 = Conv2D(16, (8, 8), activation='relu', padding='same')(input_image)
        p1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv1)
        conv2 = Conv2D(32, (4, 4), activation='relu')(p1)
        p2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(conv2)
        d1 = Dropout(0.3)(p2)
        flatten = Flatten()(d1)

        input_vector = Input(shape=(4,))
        dense1 = Dense(128, activation='relu')(input_vector)
        concat = concatenate([flatten, dense1])
        dense2 = Dense(256, activation='relu')(concat)
        d2 = Dropout(0.2)(dense2)
        critic_output = Dense(1, activation='linear')(d2)
        model = Model(inputs=[input_image, input_vector], outputs=critic_output)
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
        model.summary()
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.actor.predict(state)
        print(act_values[0], np.argmax(act_values[0]))
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        # Pobranie losowego podzbioru z pamięci
        minibatch = random.sample(self.memory, batch_size)

        # Iteracja po próbkach i aktualizacja wag sieci
        for state, action, reward, next_state, done in minibatch:
            # Obliczenie nagrody przyszłej
            next_value = self.critic.predict(next_state)[0][0]
            td_target = reward + self.gamma * next_value

            # Obliczenie błędów krytyka i aktora
            td_error = td_target - self.critic.predict(state)[0][0]
            actor_error = np.zeros((1, self.actor.output_shape[1]))
            actor_error[0][action] = td_error

            # Aktualizacja wag krytyka i aktora
            self.critic.fit(state, np.array([[td_target]]), verbose=0)
            self.actor.fit(state, actor_error, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        self.epsilon = self.epsilon_min
