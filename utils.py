import tensorflow as tf
import numpy as np
import json
from datasets import read_image_inf
from settings_train import IMAGE_SIZE
from models import ImageCaptioningModel


def get_inference_model(model_config_path):
    with open(model_config_path) as json_file:
        model_config = json.load(json_file)

    EMBED_DIM = model_config["EMBED_DIM"]
    FF_DIM = model_config["FF_DIM"]
    NUM_HEADS = model_config["NUM_HEADS"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]
    CNN_MODEL = model_config["CNN_MODEL"]
    VALUE_DIM = model_config["VALUE_DIM"]
    KEY_DIM = model_config["KEY_DIM"]
    SEQ_LENGTH = model_config["SEQ_LENGTH"]
    VOCAB_SIZE = model_config["VOCAB_SIZE"]

    # get model
    model = ImageCaptioningModel(
        cnn_model=CNN_MODEL,
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        seq_length=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
    )

    # get cnn input
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # set training to False
    training = False

    # get decoder input
    decoder_input = tf.keras.layers.Input(shape=(None,))
    model([cnn_input, training, decoder_input])
    return model


def generate_caption(image_path, model, tokenizer, SEQ_LENGTH):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # read the image from the local
    img = read_image_inf(image_path)

    # pass the image to the CNN model
    img = model.cnn_model(img)

    # pass the image features to the encoder
    encoded_img = model.encoder(img, training=False)

    # generate the caption using the decoder
    decoded_caption = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    return decoded_caption.replace("<start> ", "")


def save_tokenizer(tokenizer, path_save):
    input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    output = tokenizer(input)
    model = tf.keras.Model(input, output)
    model.save(path_save + "/tokenizer", save_format="tf")
