import tensorflow as tf

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
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    num_epochs = 2

    flag = True
    step = 0


    while step <= num_epochs and flag:

        for real_image, real_label in train_ds: pass

        if (step + 1) == num_epochs:
            flag = False
            break


if __name__ == "__main__":
    train()