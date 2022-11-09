import re
import numpy as np
import tensorflow as tf
from settings_train import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_TRAIN_IMG,
    NUM_VALID_IMG,
    SHUFFLE_DIM,
)

AUTOTUNE = tf.data.AUTOTUNE


@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    remove_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(remove_chars), "")


def add_token(list_of_list_captions):
    newLists = []
    for index, _ in enumerate(list_of_list_captions):
        new_captions = [
            "<start> {} <end>".format(caption)
            for caption in list_of_list_captions[index]
        ]
        newLists.append(new_captions)
    return newLists


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    # get the list of all image names
    all_images = list(caption_data.keys())

    # shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # return the splits
    return training_data, validation_data


def valid_test_split(captions_mapping_valid):
    valid_data = {}
    test_data = {}
    count_valid = 0
    for id in captions_mapping_valid:
        if count_valid < NUM_VALID_IMG:
            valid_data.update({id: captions_mapping_valid[id]})
            count_valid += 1
        else:
            test_data.update({id: captions_mapping_valid[id]})
            count_valid += 1
    return valid_data, test_data


def read_image_inf(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, axis=0)
    return img


def read_image(data_aug):
    def decode_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)

        if data_aug:
            img = augment(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    def augment(img):
        img = tf.expand_dims(img, axis=0)
        img = img_transform(img)
        img = tf.squeeze(img, axis=0)
        return img

    return decode_image


img_transform = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomContrast(factor=(0.05, 0.15)),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(
            height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)
        ),
        tf.keras.layers.experimental.preprocessing.RandomZoom(
            height_factor=(-0.10, 0.10), width_factor=(-0.10, 0.10)
        ),
        tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.10, 0.10)),
    ]
)


def make_dataset(images, list_of_list_captions, data_aug, tokenizer):
    read_image_xx = read_image(data_aug)
    img_dataset = tf.data.Dataset.from_tensor_slices(images)

    img_dataset = img_dataset.map(read_image_xx, num_parallel_calls=AUTOTUNE)

    list_of_list_captions_with_token = add_token(list_of_list_captions)

    cap_dataset = tf.data.Dataset.from_tensor_slices(
        list_of_list_captions_with_token
    ).map(tokenizer, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset


def reduce_dataset_dim(captions_mapping_train, captions_mapping_valid):
    train_data = {}
    count_train = 0
    for id in captions_mapping_train:
        if count_train <= NUM_TRAIN_IMG:
            train_data.update({id: captions_mapping_train[id]})
            count_train += 1
        else:
            break

    valid_data = {}
    count_valid = 0
    for id in captions_mapping_valid:
        if count_valid <= NUM_VALID_IMG:
            valid_data.update({id: captions_mapping_valid[id]})
            count_valid += 1
        else:
            break

    return train_data, valid_data
