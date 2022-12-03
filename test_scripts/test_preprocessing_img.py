import tensorflow as tf
import matplotlib.pyplot as plt


IMAGE_SIZE = [256, 256]
IMG_PATH = "images/2.jpg"

fig = plt.figure()


def read_image():
    def preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title("Sebelum Prapemrosesan", fontsize=10, fontname="Times New Roman")
        plt.imshow(img)
        print("\n\nUINT8 RESULT:", img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        print("\n\nIMG DTYPE RESULT:", img)
        img = tf.image.resize(img, IMAGE_SIZE)
        print("\n\nIMG RESIZED:", img)
        img = transform(img)
        print("\n\nIMG AUGMENTED:", img)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title("Setelah Prapemrosesan", fontsize=10, fontname="Times New Roman")
        plt.imshow(img)
        plt.show()
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
