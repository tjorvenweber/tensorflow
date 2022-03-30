import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from models.ageCGAN import Generator, Discriminator
from models.encoder import Encoder
from models.faceRec import ResNet
from load_data import load_data_from_mat, int2onehot
from losses import euclidean_dist
from optimizers import lbfgs_opt

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
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
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

    gen_losses = []
    disc_losses = []
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

                gen_losses.append(generator_loss)
                disc_losses.append(discriminator_loss)

                # optimize generator and discriminator
                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
                gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

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
                    tf.keras.utils.save_img('/notebooks/data/output0/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])


        # visualize loss
        train_G_losses.append(np.mean(gen_losses))
        train_D_losses.append(np.mean(disc_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1, = plt.plot(train_G_losses)
        line2, = plt.plot(train_D_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1,line2),("G Loss","D Loss"))
        plt.title("GAN Loss")
        plt.savefig('/notebooks/data/gan_loss.png')
        plt.close()

        if (epoch + 1) == n_epochs:
            flag = False
            break

    # save model weights
    generator.save_weights('/notebooks/models/weights/gen/gen_weights')
    discriminator.save_weights('/notebooks/models/weights/disc/disc_weights')
    

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
    # where z_i are random latent vectors
    n_pairs = 100000
    z_i = tf.random.normal([n_pairs, 100])

    # y_i are random age conditions uniformly distributed between six age categories
    n_age_cat = 6
    y_i = tf.random.uniform([n_pairs], 0, n_age_cat, dtype=tf.int64)
    y_i = tf.keras.utils.to_categorical(y_i, n_age_cat)

    # load generator weights
    generator = Generator()
    generator.load_weights('/notebooks/models/weights/gen/gen_weights')

    encoder_losses = []
    train_E_losses = []

    # train for n_epochs
    for epoch in range(n_epochs):
        print("epoch: {}".format(epoch))

        n_batches = int(z_i.shape[0] / batch_size)

        # mini-batches
        for batch in range(n_batches):
            z_batch = z_i[batch * batch_size:(batch + 1) * batch_size]
            y_batch = y_i[batch * batch_size:(batch + 1) * batch_size]

            with tf.GradientTape() as encoder_tape:
                # G(z,y) is the generator of the priorly trained Age-cGAN and x_i = G(z_i,y_i) are the synthetic face images
                x_batch = generator(z_batch, y_batch, False)
                # train encoder
                estimated_lv = encoder(x_batch, True)

                # calculate encoder loss
                encoder_loss = euclidean_dist(z_batch, estimated_lv)
                encoder_losses.append(encoder_loss)

                # optimize encoder
                encoder_gradients = encoder_tape.gradient(encoder_loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
        
            if batch % 50 == 0:
                print("batch {}".format(batch))
                print(encoder_loss)

        # visualize loss
        train_E_losses.append(np.mean(encoder_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1 = plt.plot(train_E_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1),("E Loss"))
        plt.title("Encoder Loss")
        plt.savefig('/notebooks/data/enc_loss.png')
        plt.close()

    encoder.save_weights('/notebooks/models/weights/enc/enc_weights')


def optimize_lv():
    """
    Optimize latent vector
    """
    n_epochs = 10
    learning_rate = 0.0002
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.5,
        beta_2=0.999,
        epsilon=1e-8
    )

    train_ds = load_data_from_mat()

    # load generator model and weights
    generator = Generator()
    generator.load_weights('/notebooks/models/weights/gen/gen_weights').expect_partial()

    # load encoder model and weights
    encoder = Encoder()
    encoder.load_weights('/notebooks/models/weights/enc/enc_weights').expect_partial()

    # load pretrained res net model
    res_net = ResNet()

    flag = True
    epoch = 0

    encoder_losses = []
    train_E_losses = []

    # train for n_epochs
    while epoch <= n_epochs and flag:
        print("epoch: {}".format(epoch))
        epoch += 1
        batch = 0

        for real_image, real_label in train_ds:
            batch += 1

            with tf.GradientTape() as encoder_tape:
                # get initial latent approximations z0
                z0 = encoder(real_image, True)
                
                # optimize using the L-BFGS-B algorithm
                z_opt = lbfgs_opt(generator, res_net, real_image, real_label, z0)
                z_opt = z_opt.position

                # calculate encoder loss
                encoder_loss = euclidean_dist(z_opt, z0)
                encoder_losses.append(encoder_loss)

                # optimize encoder
                encoder_gradients = encoder_tape.gradient(encoder_loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))

            if batch % 50 == 0:
                print("batch {}".format(batch))
                print(encoder_loss)

                """
                Generate and save images
                """
                image_z0 = generator(z0, real_label, False)
                image_z_opt = generator(z_opt, real_label, False)
                tf.keras.utils.save_img('/notebooks/data/output1/epoch_' + str(epoch) + '_z0.png', image_z0[0])
                tf.keras.utils.save_img('/notebooks/data/output1/epoch_' + str(epoch) + '_z_opt.png', image_z_opt[0])

        # visualize loss
        train_E_losses.append(np.mean(encoder_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1 = plt.plot(train_E_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1),("E Loss"))
        plt.title("Optimized Encoder Loss")
        plt.savefig('/notebooks/data/opt_enc_loss.png')
        plt.close()

        if (epoch + 1) == n_epochs:
            flag = False
            break

    # save model weights
    encoder.save_weights('/notebooks/models/weights/opt_enc/opt_enc_weights')


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
    elif args.model == "Optimizer":
        optimize_lv()
    else: pass
        # Not implemented error
