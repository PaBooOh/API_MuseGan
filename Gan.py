from Config import *
from Generator import Generator
from Discriminator import Discriminator
from keras import Model
from keras.models import Sequential


class Gan:
    def __init__(self, seq_length, seq_shape):
        self.seq_length = seq_length
        self.seq_shape = seq_shape

    def build_gan(self):
        generator_obj = Generator(self.seq_length, self.seq_shape)
        discriminator_obj = Discriminator(self.seq_shape)
        generator = generator_obj.buildGenerator()
        discriminator = discriminator_obj.buildDiscriminator()
        discriminator.compile(loss='binary_crossentropy', optimizer=OPT, metrics=['accuracy'])
        discriminator.trainable = False # Only train generator
        pred = discriminator(generator.output)
        gan = Model(inputs=generator.input, outputs=pred)
        gan = Sequential([generator, discriminator])
        gan.compile(loss='binary_crossentropy', optimizer=OPT)
        return generator, discriminator, gan


