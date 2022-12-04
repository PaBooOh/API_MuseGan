from keras.optimizers import Adam
"""Configuration"""

# Dataset
DATASET_PATH = 'dataset'



# MIDI
SEQUENCE_LENGTH = 100
N_MUSIC = 5



# GAN
SEQUENCE_SHAPE = (SEQUENCE_LENGTH , 1)
LATENT_DIMENSION = 1000
BATCH_SIZE = 64
OPT = Adam(learning_rate=0.0002, beta_1=0.5)
