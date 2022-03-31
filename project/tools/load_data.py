import tensorflow as tf 
import pandas as pd 
import scipy.io

from datetime import datetime, timedelta
from typing import List

# MATLAB DATA STRUCTURE: dict['imdb'] -> array[dob, photo_taken, filename, gender, name, face_location, face_score, second_face_score, celeb_name, celeb_id]
def load_data_from_mat(config):
    
    # TODO: path from config file
    # data = scipy.io.loadmat('/notebooks/data/tmp/wiki_crop/wiki.mat')['wiki'][0][0]
    # data = scipy.io.loadmat('/tmp/imdb_crop/imdb.mat')['imdb'][0][0]
    data = scipy.io.loadmat(config['data']['mat_file'])['imdb'][0][0]
    dob = data[0][0]
    photo_taken = data[1][0]
    age_labels = get_age(dob, photo_taken)

    # TODO: make prettier -> flatten
    files_array = data[2][0]
    file_names = []
    for name in files_array:
        file_names.append(name[0])

    face_score = data[6][0].tolist()
    second_face_score = data[7][0].tolist()

    df = pd.DataFrame(
        list(zip(file_names, age_labels, face_score, second_face_score)),
        columns=['file_names', 'age_labels', 'face_scores', 'second_face_scores']
    )

    data_dict = dict(zip(file_names, age_labels))
    data_dict = {k: v for k, v in data_dict.items() if k.startswith('00/')}

    # filter dataframe: second face score = NaN when no other face detected
    df = df[df['second_face_scores'].isna()]

    # filter dataframe: the higher the face score, the 'better' the face image
    good_faces = df['face_scores'] > 3
    df = df[good_faces]

    print(df.shape)

    # create dataset and prepare
    # train_ds = tf.data.Dataset.from_tensor_slices((df['file_names'], df['age_labels'])).take(12800)train_ds = tf.data.Dataset.from_tensor_slices((df['file_names'], df['age_labels'])).take(12800)
    train_ds = tf.data.Dataset.from_tensor_slices((data_dict.keys(), data_dict.values())).take(10)
    train_ds = train_ds.apply(prepare_data)

    return train_ds


def load_data_from_csv(config):

    # TODO: path from config file
    directory = config['data']['directory']
    train_df = pd.read_csv(directory + 'imdb_train_new_1024.csv')

    file_names = train_df['filename'].values
    age_labels = train_df['age'].values

    train_ds = tf.data.Dataset.from_tensor_slices((file_names, age_labels))
    train_ds = train_ds.apply(prepare_data)

    return train_ds


# TODO: batch size from config file
def prepare_data(data):
    # read image from file path
    data = data.map(read_image)
    # augment data
    data = data.map(augment)
    # run image normalization
    data = data.map(lambda img, label: (img/128 - 1, label))
    # create onehot encoded labels
    data = data.map(int2onehot)
    # cache process in memory
    data = data.cache()
    # shuffle, batch, prefetch
    data = data.shuffle(1000)
    data = data.batch(128)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)

    return data


def get_age(dob, photo_taken):

    age_labels = []
    for date_b, date_p in zip(dob, photo_taken):
        # TODO: fix
        if date_b < 100000:
            age_labels.append(-1)
        else:
            dob_datetime = datetime.fromordinal(int(date_b))  + timedelta(days=int(date_b%1)) - timedelta(days=366)
            photo_taken_datetime = datetime(date_p, 6, 15)
            age = int((photo_taken_datetime - dob_datetime).days / 365.2425)
            age_labels.append(age)

    return age_labels


def int2onehot(img, int_label):
    if int_label <= 18:
        onehot_label = tf.one_hot(0, 6)
    elif int_label > 18 and int_label <= 29:
        onehot_label = tf.one_hot(1, 6)
    elif int_label > 29 and int_label <= 39:
        onehot_label = tf.one_hot(2, 6)
    elif int_label > 39 and int_label <= 49:
        onehot_label = tf.one_hot(3, 6)
    elif int_label > 49 and int_label <= 59:
        onehot_label = tf.one_hot(4, 6)
    else:
        onehot_label = tf.one_hot(5, 6)

    return img, onehot_label


def read_image(image_file, label):
    # TODO: get path from config file
    # directory = '/notebooks/data/tmp/wiki_crop/'
    # directory = '/tmp/imdb_crop/'
    directory = './dataset/imdb/'
    file_path = directory + image_file
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)

    return image, label


def augment(img, label): 
    image = tf.image.resize(img, [128, 128])
    return image, label
