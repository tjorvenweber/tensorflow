import tensorflow as tf
import argparse
import numpy as np

from models.ageCGAN import Generator
from models.encoder import Encoder
from tools.load_data import int2onehot


def change_age(image_path, age):

    # read image
    image = load_image(image_path)

    # resize and expand dimension to be processable by the encoder
    image = tf.image.resize(image, [128, 128])
    image = tf.expand_dims(image, 0)

    # set up models
    encoder = Encoder()
    encoder.load_weights('./weights/enc/enc_weights').expect_partial()

    generator = Generator()
    generator.load_weights('./weights/gen/gen_weights').expect_partial()

    # compress identity into z
    z_identity = encoder(image, False)

    # generate face image in the given age category
    _, age_label = int2onehot(None, age)
    age_label = [age_label]
    modified_image = generator(z_identity, age_label, False)

    # save modified image to output folder
    tf.keras.utils.save_img('./output/image_age_{}.png'.format(age), modified_image[0]) 

def load_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image
    except:
        print("ERROR: Image could not be opened!")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="face_aging")
    parser.add_argument(
        "--image",
        nargs="?",
        type=str,
        default=None,
        help="Path to image"
    )
    parser.add_argument(
        "--age",
        nargs="?",
        type=int,
        default=42,
        help="Age to apply to image"
    )

    args = parser.parse_args()

    if (args.image == None):
        print("ERROR: No image path!")

    change_age(args.image, args.age)