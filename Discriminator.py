from Config import *
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, LeakyReLU


class Discriminator:
    
    def __init__(self, input_shape=SEQUENCE_SHAPE):
        self.input_shape = input_shape

    def buildDiscriminator(self):
        discriminator = Sequential(
            [
                LSTM(512, input_shape=self.input_shape, return_sequences=True),
                Bidirectional(LSTM(512)),
                Dense(512),
                LeakyReLU(alpha=0.2),
                Dense(1024),
                LeakyReLU(alpha=0.2),
                Dense(1, activation='sigmoid')
            ]
        )
        return discriminator