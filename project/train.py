import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import yaml

from models.ageCGAN import Generator, Discriminator
from models.encoder import Encoder
from models.faceRec import ResNet
from tools.load_data import load_data_from_mat, int2onehot
from tools.losses import euclidean_dist, disc_label_flipping, gen_vanilla
from tools.optimizers import lbfgs_opt, get_optimizer


def train_GAN(config):

    """
    Load data
    """

    train_ds = load_data_from_mat(config)

    """
    Set up optimizer for generator and discriminator
    """
    g_optimizer_class = get_optimizer(config, 'generator')
    g_optimizer_params = {
        k: v for k, v in config['optimizer']['generator'].items() if k != 'name'}
    g_optimizer = g_optimizer_class(**g_optimizer_params)

    d_optimizer_class = get_optimizer(config, 'discriminator')
    d_optimizer_params = {
        k: v for k, v in config['optimizer']['discriminator'].items() if k != 'name'}
    d_optimizer = d_optimizer_class(**d_optimizer_params)


    """
    Set up generator and discriminator models
    """
    generator, discriminator = Generator(), Discriminator()

    fixed_noise = tf.random.normal([config['training']['batch_size'], config['training']['z_dimension']])
    random_age = np.random.randint(low=0, high=100, size=config['training']['batch_size'])
    fixed_labels = [int2onehot(None, age)[1] for age in random_age]
    epoch = 0

    """
    Train generator and discriminator
    """

    G_losses = []
    D_losses = []
    train_G_losses = []
    train_D_losses = []

    while epoch <= config['training']['epochs']:
        print("epoch: {}".format(epoch))
        epoch += 1
        batch = 0

        for real_image, real_label in train_ds:
            batch += 1

            z = tf.random.normal([config['training']['batch_size'], config['training']['z_dimension']])
            random_age = np.random.randint(low=0, high=100, size=config['training']['batch_size'])
            fake_label = [int2onehot(None, age)[1] for age in random_age]

            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                # get predictions
                fake_image = generator(z, fake_label, True)
                fake_features, fake_data_pred = discriminator(fake_image, fake_label, True)
                real_features, real_data_pred = discriminator(real_image, real_label, True)
              
                # get loss with label flipping
                discriminator_loss = disc_label_flipping(discriminator, real_data_pred, fake_data_pred)
                generator_loss = gen_vanilla(generator, fake_data_pred)

                G_losses.append(generator_loss)
                D_losses.append(discriminator_loss)

                # optimize generator and discriminator
                generator_gradients = generator_tape.gradient(generator_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
                discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

            if batch % config['training']['update_interval'] == 0:
                print("batch {}".format(batch))
                print("Discriminator Loss: {}, Generator Loss: {}".format(discriminator_loss, generator_loss))

                """
                Generate and save images
                """

                # visualize images
                images = generator(fixed_noise, fixed_labels, False)
                for i in range(5):
                    # tf.keras.utils.save_img('./output/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])
                    # tf.keras.utils.save_img('/content/drive/MyDrive/faceAging/images/0/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])
                    tf.keras.utils.save_img(config['data']['output_path'] + 'images/gan/epoch_' + str(epoch) + '_' + str(i) + '.png', images[i])


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
        plt.savefig(config['data']['output_path'] + 'plots/gan_loss.png')
        plt.close()
        

        # save model weights
        generator.save_weights(config['data']['weight_path'] + 'gen/gen_weights')
        discriminator.save_weights(config['data']['weight_path'] + 'disc/disc_weights')




def train_encoder(config):
    """
    Set up optimizer
    """
    
    optimizer_class = get_optimizer(config, 'encoder')
    optimizer_params = {
        k: v for k, v in config['optimizer']['encoder'].items() if k != 'name'}
    optimizer = optimizer_class(**optimizer_params)

    """
    Set up encoder model
    """
    encoder = Encoder()

    """
    Train encoder
    """
    # to train E, we generate a synthetic dataset of 100K pairs (x_i,G(z_i,y_i)), i = 1,...,10^5
    # where z_i are random latent vectors
    # n_pairs = 100000
    n_pairs = 10
    z_i = tf.random.normal([n_pairs, config['training']['z_dimension']])

    # y_i are random age conditions uniformly distributed between six age categories
    n_age_cat = 6
    y_i = tf.random.uniform([n_pairs], 0, n_age_cat, dtype=tf.int64)
    y_i = tf.keras.utils.to_categorical(y_i, n_age_cat)

    # load generator weights
    generator = Generator()
    # generator.load_weights(config['data']['weight_path'] + 'gen/gen_weights')
    generator.load_weights('exp_weights/gen/gen_weights').expect_partial()

    encoder_losses = []
    train_E_losses = []

    # train for n_epochs
    for epoch in range(config['training']['epochs']):
        print("epoch: {}".format(epoch))

        n_batches = int(z_i.shape[0] / config['training']['batch_size'])

        # mini-batches
        for batch in range(n_batches):
            z_batch = z_i[batch * config['training']['batch_size']:(batch + 1) * config['training']['batch_size']]
            y_batch = y_i[batch * config['training']['batch_size']:(batch + 1) * config['training']['batch_size']]

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
        
            if batch % config['training']['update_interval'] == 0:
                print("batch {}".format(batch))
                print("Encoder Loss: {}".format(encoder_loss))

        # visualize loss
        train_E_losses.append(np.mean(encoder_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1 = plt.plot(train_E_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1),("E Loss"))
        plt.title("Encoder Loss")
        plt.savefig(config['data']['output_path'] + 'plots/gan_loss.png')
        plt.close()

    encoder.save_weights(config['data']['weight_path'] + 'enc/enc_weights')


def optimize_lv(config):
    """
    Optimize latent vector
    """

    optimizer_class = get_optimizer(config, 'z_optimizer')
    optimizer_params = {
        k: v for k, v in config['optimizer']['z_optimizer'].items() if k != 'name'}
    optimizer = optimizer_class(**optimizer_params)


    train_ds = load_data_from_mat(config)

    # load generator model and weights
    generator = Generator()
    generator.load_weights(config['data']['weight_path'] + 'gen/gen_weights').expect_partial()

    # load encoder model and weights
    encoder = Encoder()
    encoder.load_weights(config['data']['weight_path'] + 'enc/enc_weights').expect_partial()

    # load pretrained res net model
    res_net = ResNet()


    epoch = 0
    encoder_losses = []
    train_E_losses = []

    # train for n_epochs
    while epoch <= config['training']['epochs']:
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
                tf.keras.utils.save_img(config['data']['output_path'] + 'images/z_optimizer/epoch_' + str(epoch) + '_z0.png', image_z0[0])
                tf.keras.utils.save_img(config['data']['output_path'] + 'images/z_optimizer/epoch_' + str(epoch) + '_z_opt.png', image_z_opt[0])



        # visualize loss
        train_E_losses.append(np.mean(encoder_losses))

        # TODO: outsource/tensorboard
        plt.figure()
        line1 = plt.plot(train_E_losses)
        plt.xlabel("Training steps")
        plt.ylabel("Loss")
        plt.legend((line1),("E Loss"))
        plt.title("Optimized Encoder Loss")
        plt.savefig(config['data']['output_path'] + 'plots/opt_enc_loss.png')
        plt.close()


    # save model weights
    encoder.save_weights(config['data']['weight_path'] + 'opt_enc/opt_enc_weights')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        default="AgeCGAN",
        help="Model to train"
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/ageCGAN.yml",
        help="Model to train"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    if args.model == "AgeCGAN":
        train_GAN(config)
    elif args.model == "Encoder":
        train_encoder(config)
    elif args.model == "Optimizer":
        optimize_lv(config)
    else:
        print("No valid model found")
