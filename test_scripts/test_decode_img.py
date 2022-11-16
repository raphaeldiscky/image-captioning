import tensorflow as tf


IMAGE_SIZE = [256, 256]
IMG_PATH = "images/1.jpg"


def read_image(data_aug):
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        print("\n\nUINT8:", img)
        img = tf.image.resize(img, IMAGE_SIZE)
        print("\n\nIMG RESIZE:", img)
        print("SHAPE", tf.shape(img))
        if data_aug:
            img = augment(img)
        print("\n\nAUGMENT:", img)
        img = tf.image.convert_image_dtype(img, tf.float32) / 255.0
        print("\n\nIMG DTYPE:", img)
        print("SHAPE FINAL", tf.shape(img))
        return img

    def augment(img):
        img = tf.expand_dims(img, axis=0)
        img = transform(img)
        img = tf.squeeze(img, axis=0)
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

read_image(data_aug=True)
