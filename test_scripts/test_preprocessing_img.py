import tensorflow as tf


IMAGE_SIZE = [256, 256]
IMG_PATH = "images/1.jpg"


def read_image():
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        print("\n\nUINT8 RESULT:", img)
        img = tf.image.resize(img, IMAGE_SIZE)
        print("\n\nIMG RESIZED:", img)
        img = transform(img)
        print("\n\nIMG AUGMENTED:", img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        print("\n\nIMG DTYPE RESULT:", img)
        return img

    return preprocess_image(IMG_PATH)


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

read_image()
