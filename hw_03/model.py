import tensorflow as tf
from tensorflow.keras.layers import Dense

class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        #hidden layer 1: 256 units with sigmoid activation function
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)
        #hidden layer 2: 256 units with sigmoid activation function
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid)
        #output: 10 units with softmax activation function
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x