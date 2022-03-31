from re import T
import tensorflow as tf 

from models.ageCGAN import Generator
from models.encoder import Encoder

"""
Model that recognizes a person’s identity in an input face image
"""
class ResNet(tf.keras.Model):

  def __init__(self):
    super(ResNet, self).__init__()

    self.res_net = tf.keras.applications.vgg16.VGG16(
      include_top=False,
      weights="imagenet",
      pooling='avg'
    )

    self.out = tf.keras.layers.Dense(units=128)

  def call(self, x, training):

    x = self.res_net(x)
    x = self.out(x)

    return x

