import tensorflow as tf
import numpy as np
import json
from datasets import read_image_inf
from settings_train import IMAGE_SIZE
from models import (
    get_cnn_model,
    Decoder,
    Encoder,
    ImageCaptioningModel,
)


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

    # get model
    cnn_model = get_cnn_model(CNN_MODEL)
    encoder = Encoder(
        embed_dim=EMBED_DIM,
        ff_dim=FF_DIM,
        num_heads=NUM_HEADS,
        key_dim=KEY_DIM,
        value=VALUE_DIM,
    )
    decoder = Decoder(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        vocab_size=VOCAB_SIZE,
        key_dim=KEY_DIM,
        VALUE_DIM=VALUE_DIM,
    )
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder
    )

    # get cnn input
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # set training to False
    training = False

    # get decoder input
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input, training, decoder_input])
    return caption_model


def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH):
    vocab = tokenizer.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    # read the image from the local
    img = read_image_inf(image_path)

    # pass the image to the CNN model
    img = caption_model.cnn_model(img)

    # pass the image features to the encoder
    encoded_img = caption_model.encoder(img, training=False)

    # generate the caption using the decoder
    decoded_caption = "<start>"
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
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
