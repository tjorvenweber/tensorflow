import tensorflow as tf 

# TODO: add typing

"""
Generator that creates synthetic images
"""
class Generator(tf.keras.Model):

    def __init__(self) -> None:
        super(Generator, self).__init__()

        dropout_rate = 0.2
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # TODO: create blocks to remove duplicate code
        self.gen = [
            # 4 x 4 x 1024
            tf.keras.layers.Dense(units=(4*4*2048)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Reshape((4, 4, 2048)),

            # 8 x 8 x 512
            tf.keras.layers.Conv2DTranspose(1024, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(center=1.0, scale=0.02),
            tf.keras.layers.Activation(tf.nn.relu),

            # 16 x 16 x 256
            tf.keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(center=1.0, scale=0.02),
            tf.keras.layers.Activation(tf.nn.relu),

            # 32 x 32 x 128
            tf.keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(center=1.0, scale=0.02),
            tf.keras.layers.Activation(tf.nn.relu),

            # 64 x 64 x 64
            tf.keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(center=1.0, scale=0.02),
            tf.keras.layers.Activation(tf.nn.relu),

            # 128 x 128 x 3
            tf.keras.layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.Activation(tf.nn.tanh)
        ]


    def __call__(self, x, y, training_flag):

        # concatenate age label and input
        x = tf.concat([x, y], axis=1)

        for gen_layer in self.gen:

            if isinstance(gen_layer, tf.keras.layers.BatchNormalization):
                x = gen_layer(x, training_flag)
            else:
                x = gen_layer(x)

        return x


"""
Discriminator that distinguishes real face images from synthetic images produced by the generator
"""
class Discriminator(tf.keras.Model):

    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)

        # TODO: create blocks to remove duplicate code
        self.disc = [
            # 64 x 64 x 64
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.LeakyReLU(0.2),

            # 32 x 32 x 128
            tf.keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            # 16 x 16 x 256
            tf.keras.layers.Conv2D(256, kernel_size=4, strides=2, padding="same", kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            # 8 x 8 x 512
            # tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(0.2),

            # 4 x 4 x 1024
            # tf.keras.layers.Conv2D(1024, kernel_size=3, strides=2, padding="same", kernel_initializer=initializer),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.LeakyReLU(0.2),

            # 1 x 1 x 1
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=1),
            #tf.keras.layers.Conv2D(1, kernel_size=4, strides=2, padding="valid"),
            tf.keras.layers.Activation(tf.nn.sigmoid)
        ]


    def __call__(self, x, y, training_flag):


        # TODO: make prettier, outsource

        # concatenate age label to first conv layer
        y_exp = tf.keras.backend.expand_dims(y, -1)
        y_exp = tf.keras.backend.expand_dims(y_exp, -1)
        y_tiled = tf.keras.backend.tile(y_exp, [1, 21, 128, 1])

        y_rest = [y_label[:2] for y_label in y]
        y_rest = tf.keras.backend.expand_dims(y_rest, -1)
        y_rest = tf.keras.backend.expand_dims(y_rest, -1)
        y_rest = tf.keras.backend.tile(y_rest, [1, 1, 128, 1])

        label = tf.concat([y_tiled, y_rest], axis=1)
        x = tf.concat([x, label], axis=-1)
        

        for disc_layer in self.disc:

            if isinstance(disc_layer, tf.keras.layers.BatchNormalization):
                x = disc_layer(x, training_flag)
            else:
                x = disc_layer(x)

        return x
