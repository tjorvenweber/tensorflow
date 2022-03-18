import tensorflow as tf
import matplotlib.pyplot as plt

from models.AgeCGAN import Generator, Discriminator
from load_data import load_data_from_mat

# TODO: document command line parameters in README
# TODO: method for saving trained model


# TODO
def init_weights(): pass



# TODO: config, writer, logger, args
def train():

    # TODO: set up seeds, device (cuda), augmentations (e.g. cropping/resizing)
    # TODO: set up logging/tensorboard
    # TODO: set up metrics
    # TODO: outsource optimizers and loss + get params from config

    # load data
    train_ds = load_data_from_mat()

    # set up model 
    generator, discriminator = Generator(), Discriminator()

    # set up optimizer
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    num_epochs = 2
    train_iters = 50

    flag = True
    step = 0

    fixed_noise = tf.random.normal([4, 10])

    # visualize
    test_data = train_ds.take(1)
    test_images = []
    for image, label in test_data:
        test_images.append(image[0, :, :, :])
        test_images.append(image[1, :, :, :])
        test_images.append(image[2, :, :, :])
        test_images.append(image[3, :, :, :])

    plt.figure(figsize=(25, 25))

    for i in range(len(test_images)):
        ax1 = plt.subplot(2, 10, i+1)
        plt.imshow(tf.squeeze(test_images[i]))
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

    plt.show()


    while step <= train_iters and flag:

        print("step: {}".format(step))

        for real_image, real_label in train_ds:
            step += 1

            # TODO: Batchsize, z dimension from config
            # TODO: Batchsize and z_dim in paper
            z = tf.random.normal([4, 10])
            # TODO: age label

            with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
                fake_image = generator(z, True)
                fake_data_pred = discriminator(fake_image, True)
                real_data_pred = discriminator(real_image, True)

                D_loss = -tf.math.reduce_mean(tf.math.log(real_data_pred) + tf.math.log(1 - fake_data_pred))
                G_loss = tf.math.reduce_mean(tf.math.log(1 - fake_data_pred))
                # print(D_loss, G_loss)

                D_gradients = D_tape.gradient(D_loss, discriminator.trainable_variables)
                optimizer.apply_gradients(zip(D_gradients, discriminator.trainable_variables))

                G_gradients = G_tape.gradient(G_loss, generator.trainable_variables)
                optimizer.apply_gradients(zip(G_gradients, generator.trainable_variables))


        # visualize
        images = generator(fixed_noise, False)
        plt.figure(figsize=(25, 25))

        for i in range(len(images)):
            ax1 = plt.subplot(2, 10, i+1)
            plt.imshow(tf.squeeze(images[i]))
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

        plt.show()


        if (step + 1) == train_iters:
            flag = False
            break


if __name__ == "__main__":
    train()