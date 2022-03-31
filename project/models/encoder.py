import tensorflow as tf 

"""
Encoder that encodes an image to latent vector z
"""
class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__()

    self.enc_layers = [
            tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Conv2D(256, kernel_size=(5,5), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(4096),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.nn.relu),

            tf.keras.layers.Dense(units=100)
    ]

  def call(self, x, training):
    for layer in self.enc_layers:

      if (isinstance(layer, tf.keras.layers.BatchNormalization)):
        x = layer(x, training)
      else:
        x = layer(x)

    return x

