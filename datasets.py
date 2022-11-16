import re
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


def add_token(captions):
    newLists = []
    for index, _ in enumerate(captions):
        new_captions = [
            "<start> {} <end>".format(caption) for caption in captions[index]
        ]
        newLists.append(new_captions)
    return newLists


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
    img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    return img


def read_image(data_aug):
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)

        if data_aug:
            img = augment(img)
        img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
        return img

    def augment(img):
        img = tf.expand_dims(img, axis=0)
        img = transform(img)
        img = tf.squeeze(img, axis=0)
        return img

    return preprocess_image


transform = tf.keras.Sequential(
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


def make_dataset(images, captions, data_aug, tokenizer):
    read_image_output = read_image(data_aug)
    images_dataset = tf.data.Dataset.from_tensor_slices(images)
    images_dataset = images_dataset.map(read_image_output, num_parallel_calls=AUTOTUNE)
    # add token <start> and <end> to list of captions
    data_cap_with_token = add_token(captions)
    caption_dataset = tf.data.Dataset.from_tensor_slices(data_cap_with_token).map(
        tokenizer, num_parallel_calls=AUTOTUNE
    )
    dataset = tf.data.Dataset.zip((images_dataset, caption_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(SHUFFLE_DIM).prefetch(AUTOTUNE)
    return dataset


def reduce_dataset(captions_mapping_train, captions_mapping_valid):
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
