import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from tools.models.ageCGAN import Generator, Discriminator
# from tools.models.encoder import Encoder
from tools.load_data import load_data_from_mat, int2onehot
# from tools.losses import euclidean_dist

# TODO: document command line parameters in README

def train_GAN():

    # TODO: params: config, writer, logger, args
    # TODO: set up logging/tensorboard
    # TODO: set up metrics
    # TODO: get optimizer params etc. from config

    """
    Load data
    """
    # TODO test dataset 
    train_ds = load_data_from_mat()

    """
    Hyperparameters for generator and discriminator
    """
    learning_rate = 0.0002
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    n_epochs = 100

    """
    Set up generator and discriminator models
    """
    generator, discriminator = Generator(), Discriminator()

    fixed_noise = tf.random.normal([50, 100])
    # fixed_labels = [int2onehot(None, y)[1] for y in [0, 1, 2, 3]] 
    random_age = random.sample(range(0, 61), 50)
    fixed_labels = [int2onehot(None, age)[1] for age in random_age]
    flag = True
    epoch = 0

    """
    Train generator and discriminator
    """

    G_losses = []
    D_losses = []
    train_G_losses = []
    train_D_losses = []

    while epoch <= n_epochs and flag:
        print("epoch: {}".format(epoch))
        epoch += 1
        batch = 0

        for real_image, real_label in train_ds:
            batch += 1

            # TODO: Batchsize, z dimension from config
            # TODO: make prettier
            z = tf.random.normal([50, 100])
            random_age = random.sample(range(0, 61), 50)
            fake_label = [int2onehot(None, age)[1] for age in random_age]
            '''for age in random_age:
                fake_label.append(int2onehot(None, age)[1])'''

            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                # get predictions
                fake_image = generator(z, fake_label, True)
                fake_data_pred = discriminator(fake_image, fake_label, True)
                real_data_pred = discriminator(real_image, real_label, True)

                # calculate generator and discriminator loss
                generator_loss = generator.loss_function(tf.ones_like(fake_data_pred), fake_data_pred)
                discriminator_loss = (discriminator.loss_function(tf.ones_like(real_data_pred), real_data_pred) + discriminator.loss_function(tf.zeros_like(fake_data_pred), fake_data_pred)) / 2
                #discriminator_loss = -tf.math.reduce_mean(tf.math.log(real_data_pred) + tf.math.log(1 - fake_data_pred))
                #generator_loss = tf.math.reduce_mean(tf.math.log(1 - fake_data_pred))

                G_losses.append(generator_loss)
                D_losses.append(discriminator_loss)

                # optimize generator and discriminator
                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
                optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            if batch % 50 == 0:
                print("batch {}".format(batch))
                print(discriminator_loss, generator_loss)

                """
                Generate and save images
                """
            # if batch % 50 == 0:
            #     random_age = random.sample(range(0, 61), 50)
            #     fake_label = []
            #     for age in random_age:
            #         fake_label.append(int2onehot(None, age)[1])

                # visualize images
                images = generator(fixed_noise, fixed_labels, False)
                for i in range(5):
                    # tf.keras.utils.save_img('./output/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])
                    tf.keras.utils.save_img('/content/drive/MyDrive/faceAging/images/0/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])


        # visualize loss
        train_G_losses.append(np.mean(G_losses))
        train_D_losses.append(np.mean(D_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1, = plt.plot(train_G_losses)
        line2, = plt.plot(train_D_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1,line2),("G Loss","D Loss"))
        plt.title("GAN Loss")
        # plt.savefig('./output/loss.png')
        plt.savefig('/content/drive/MyDrive/faceAging/loss_plots/loss.png')
        plt.close()
        

        # save model weights
        # generator.save_weights('./weights/gen_weights')
        # discriminator.save_weights('./weights/disc_weights')


        if (epoch + 1) == n_epochs:
            flag = False
            break

    

def train_encoder():
    """
    Hyperparameters for encoder
    """
    n_epochs = 25
    batch_size = 64
    learning_rate = 0.0002
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.5,
        beta_2=0.999,
        epsilon=1e-8
    )

    """
    Set up encoder model
    """
    encoder = Encoder()

    """
    Train encoder
    """
    # to train E, we generate a synthetic dataset of 100K pairs (x_i,G(z_i,y_i)), i = 1,...,10^5
    # where zi are random latent vectors
    n_pairs = 100000
    z_i = tf.random.normal([n_pairs, 100])

    # yi are random age conditions uniformly distributed between six age categories
    n_age_cat = 6
    y_i = tf.random.uniform([n_pairs], 0, n_age_cat, dtype=tf.int64)
    y_i = tf.keras.utils.to_categorical(y_i, n_age_cat)

    # load generator weights
    generator = Generator()
    generator.load_weights('./weights/gen_weights')

    # train for n_epochs
    for epoch in range(n_epochs):
        print("epoch: {}".format(epoch))

        encoder_losses = []
        n_batches = int(z_i.shape[0] / batch_size)

        # mini-batches
        for batch in range(n_batches):
            print("batch: {}".format(batch))

            z_batch = z_i[batch * batch_size:(batch + 1) * batch_size]
            y_batch = y_i[batch * batch_size:(batch + 1) * batch_size]

            with tf.GradientTape() as encoder_tape:
                # G(z,y) is the generator of the priorly trained Age-cGAN and xi = G(zi,yi) are the synthetic face images
                x_batch = generator(z_batch, y_batch, False)
                # train encoder
                estimated_latent_vec = encoder(x_batch, True)

                # calculate encoder loss
                encoder_loss = euclidean_dist(z_batch, estimated_latent_vec)
                encoder_losses.append(encoder_loss)
                print("Encoder loss:", encoder_loss)

                # optimize encoder
                encoder_gradients = encoder_tape.gradient(encoder_loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="AgeCGAN",
        help="Model to train"
    )
    args = parser.parse_args()
    if args.model == "AgeCGAN":
        train_GAN()
    elif args.model == "Encoder":
        train_encoder()
    else: pass
        # Not implemented error
