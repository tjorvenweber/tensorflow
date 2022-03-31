import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from tools.losses import euclidean_dist


key2opt = {
    "adam": Adam
}


def get_optimizer(config, model):
    if config['optimizer'] is None:
        return Adam

    else:
        opt_name = config['optimizer'][model]['name']
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        return key2opt[opt_name]

def lbfgs_opt(generator, res_net, real_image, real_label, z0):

    def eucl_dist_opt(z):
        # generate synthetic image given latent vector
        fake_image = generator(z, real_label, False)
        # get identity of the reconstructed image
        id_fake = res_net(fake_image, False)

        # get identity of the original image
        id_real = res_net(real_image, False)

        # calculate euclidean distance between both identities
        return euclidean_dist(id_real, id_fake)

    """
    minimze the difference between the identities in the original and reconstructed image
    """
    return tfp.optimizer.lbfgs_minimize(
        lambda z: tfp.math.value_and_gradient(eucl_dist_opt, z),
        initial_position=z0,
        tolerance=1e-8,
        max_iterations=5
    )

