from Config import *
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, BatchNormalization, Reshape, Dense, LeakyReLU


class Generator:
    
    def __init__(self, seq_length=SEQUENCE_LENGTH, output_shape=SEQUENCE_SHAPE):
        self.seq_length = seq_length
        self.output_shape = output_shape
    
    def buildGenerator(self):
        generator = Sequential(
        [
            LSTM(512, input_shape=(LATENT_DIMENSION, 1), return_sequences=True),
            Bidirectional(LSTM(512)),
            Dense(256),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(512),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(1024),
            LeakyReLU(alpha=0.2),
            BatchNormalization(momentum=0.8),
            Dense(self.seq_length, activation='tanh'),
            Reshape(self.output_shape)

        ]
    )
        return generator
    #     self.lstm = LSTM(512, input_shape=input_shape, return_sequences=True)
    #     self.bi = Bidirectional(LSTM(512))
    #     self.dense1 = Dense(256)
    #     self.acti_lrelu1 = LeakyReLU(alpha=0.2)
    #     self.bn1 = BatchNormalization(momentum=0.8)
    #     self.dense2 = Dense(512)
    #     self.acti_lrelu2 = LeakyReLU(alpha=0.2)
    #     self.bn2 = BatchNormalization(momentum=0.8)
    #     self.dense3 = Dense(1024)
    #     self.acti_lrelu3 = LeakyReLU(alpha=0.2)
    #     self.bn3 = BatchNormalization(momentum=0.8)
    #     self.dense4 = Dense(SEQUENCE_LENGTH, activation='tanh')
    #     self.reshape = Reshape(SEQUENCE_SHAPE)


    # def call(self, inputs):
    #     x = self.lstm(inputs)
    #     x = self.bi(x)
    #     x = self.dense1(x)
    #     x = self.acti_lrelu1(x)
    #     x = self.bn1(x)
    #     x = self.dense2(x)
    #     x = self.acti_lrelu2(x)
    #     x = self.bn2(x)
    #     x = self.dense3(x)
    #     x = self.acti_lrelu3(x)
    #     x = self.bn3(x)
    #     x = self.dense4(x)
    #     x = self.reshape(x)
    #     return x