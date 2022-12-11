from keras.optimizers import Adam

"""Configuration"""

# Dataset
DATASET_PATH = 'dataset'

# MIDI
SEQUENCE_LENGTH = 100
N_MUSIC = 3 # How many music we want model to generate
OUTPUT_SHAPE = (100, 1)

# GAN
SEQUENCE_SHAPE = (SEQUENCE_LENGTH , 1)
LATENT_DIMENSION = 100
BUILT_WITH_BILSTM = False # Normal LSTM, if false.

# Train
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 64
TRAIN_LEARNING_RATE = 0.0001

OPT = Adam(learning_rate=TRAIN_LEARNING_RATE)