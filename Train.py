import numpy as np
from Gan import *
from Config import *
from DataProcessing import makeData, seq_to_midi


def generateNoises(batch_size, latent_dim):
    # Generate a batch of Gaussian noise from latent space
    return np.random.randn(batch_size, latent_dim)

def getRealData(dataset, batch_size):
    random_indice = np.random.randint(0, dataset.shape[0], batch_size)
    X_real = dataset[random_indice] # seqs
    Y_real = np.ones((batch_size, 1))
    return X_real, Y_real

def getFakeData(generator, latent_dim, batch_size):
    noise = generateNoises(batch_size, latent_dim)
    X_fake = generator.predict(noise)
    Y_fake = np.zeros((batch_size, 1))
    return X_fake, Y_fake

def getBatchData(generator, latent_dim, dataset, batch_size):
    # Concatenate and merge fake data with real data
    X_real, Y_real = getRealData(dataset, batch_size)
    X_fake, Y_fake = getFakeData(generator, latent_dim, batch_size)
    X = np.concatenate([X_real, X_fake], axis=0)
    Y = np.concatenate([Y_real, Y_fake], axis=0)
    return X, Y


def train(dataset_path=DATASET_PATH, latent_dim=LATENT_DIMENSION, epochs=20, batch_size=32):
        from tqdm import tqdm
        dataset, notes_to_ints = makeData(dataset_path)
        gan_obj = Gan(SEQUENCE_LENGTH, SEQUENCE_SHAPE)
        generator, discriminator, gan = gan_obj.build_gan()

        for epoch in tqdm(range(epochs)):
            X_real, Y_real = getRealData(dataset, batch_size)
            X_fake, Y_fake = getFakeData(generator, latent_dim, batch_size)
            # X_seqs, Y_seqs = getBatchData(generator, latent_dim, dataset, batch_size)
            discriminator_loss_real = discriminator.train_on_batch(X_real, Y_real)
            discriminator_loss_fake = discriminator.train_on_batch(X_fake, Y_fake)
            d_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
            X_noises = generateNoises(batch_size, latent_dim)
            Y_noises = np.ones([batch_size, 1])
            generator_loss = gan.train_on_batch(X_noises, Y_noises)
            print("| Epoch: %d | Discriminator loss: %.3f, accuracy: %.2f%% | Generator loss: %.3f |" % (epoch, d_loss[0], 100*d_loss[1], generator_loss))
        
        # Generate several pieces of music
        generateMusic(generator, notes_to_ints, N_MUSIC)


def generateMusic(generator, notes_to_ints, n_pieces=5):
    boundary = int(len(notes_to_ints) / 2)
    for i in range(n_pieces):
        noise = generateNoises(1, LATENT_DIMENSION)
        pred_seq = generator.predict(noise)[0]
        pred_ints = [x * boundary + boundary for x in pred_seq]
        notes = [key for key in notes_to_ints]
        pred_notes = [notes[int(x)] for x in pred_ints]
        seq_to_midi(pred_notes, 'outputs/generated_' + str(i))


train(dataset_path=DATASET_PATH, latent_dim=LATENT_DIMENSION, epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE)