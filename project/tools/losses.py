import tensorflow as tf
import random
import numpy as np

def euclidean_dist(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))


def gen_vanilla(generator, fake_data_pred):
    return generator.loss_function(tf.ones_like(fake_data_pred), fake_data_pred)


def disc_vanilla(discriminator, real_data_pred, fake_data_pred):
    return discriminator.loss_function(tf.ones_like(real_data_pred), real_data_pred) + \
        discriminator.loss_function(tf.zeros_like(fake_data_pred), fake_data_pred)


def disc_label_flipping(discriminator, real_data_pred, fake_data_pred):
    if random.random() > 0.2:   
        return discriminator.loss_function(tf.ones_like(real_data_pred), real_data_pred) + \
            discriminator.loss_function(tf.zeros_like(fake_data_pred), fake_data_pred) 
    else:
        return discriminator.loss_function(tf.zeros_like(real_data_pred), real_data_pred) + \
            discriminator.loss_function(tf.ones_like(fake_data_pred), fake_data_pred) 


def gen_feature_matching(generator, fake_data_pred, real_features, fake_features):
    loss_vanilla = gen_vanilla(generator, fake_data_pred)

    data_moments = tf.reduce_mean(real_features, axis=0)
    sample_moments = tf.reduce_mean(fake_features, axis=0)
    loss_feature = tf.reduce_mean(tf.square(data_moments-sample_moments))

    return loss_vanilla + loss_feature


def gen_wasserstein(fake_data_pred):
    """
    Wasserstein Loss for Generator 
    https://arxiv.org/abs/1701.07875
    """
    with tf.compat.v1.name_scope(None, 'generator_wasserstein_loss', 
        (fake_data_pred, 1.0)) as scope:

        generator_loss = - fake_data_pred
        generator_loss = tf.compat.v1.losses.compute_weighted_loss(
            generator_loss, 
            1.0, 
            None, 
            tf.compat.v1.GraphKeys.LOSSES, 
            tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    return generator_loss


def disc_wasserstein(real_data_pred, fake_data_pred):
    """
    Wasserstein Loss for Generator 
    https://arxiv.org/abs/1701.07875
    """
    with tf.compat.v1.name_scope(None, 'discriminator_wasserstein_loss',
        (real_data_pred, fake_data_pred, 1.0, 1.0)) as scope:

        loss_on_fake = tf.compat.v1.losses.compute_weighted_loss(
            fake_data_pred,
            1.0,
            None,
            loss_collection=None,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        loss_on_real = tf.compat.v1.losses.compute_weighted_loss(
            real_data_pred,
            1.0,
            None,
            loss_collection=None,
            reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        discriminator_loss = loss_on_fake - loss_on_real
        tf.compat.v1.losses.add_loss(discriminator_loss, tf.compat.v1.GraphKeys.LOSSES)

    return discriminator_loss


def disc_wasserstein_gp(discriminator, real_data_pred, fake_data_pred, real_image, fake_image, real_label, weight_factor):
    """
    Wasserstein Loss for Discriminator with Gradient Penalty
    """
    wasserstein_loss = disc_wasserstein(real_data_pred, fake_data_pred)
    gradient_penalty = gradient_penalty(discriminator, real_image, fake_image, real_label)
    return wasserstein_loss + weight_factor * gradient_penalty


def gradient_penalty(disc, real_samples, fake_samples, labels):
    
    # random weight term for interpolation between fake and real images
    alpha = np.random.random((real_samples.shape[0], 1, 1, 1))

    # get random interpolation between real and fake samples
    x_hat = alpha * real_samples + ((1 - alpha) * fake_samples)
    
    # gradient
    with tf.GradientTape() as t:
        t.watch(x_hat)
        d_hat = disc(x_hat, labels, True)
    gradients = t.gradient(d_hat, x_hat)
    
    g_norm2 = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    d_regularizer = tf.reduce_mean((g_norm2 - 1.0) ** 2)

    return d_regularizer
    
        
