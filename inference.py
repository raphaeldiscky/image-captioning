from settings_inference import (
    MODEL_CONFIG_PATH,
    MODEL_WEIGHT_PATH,
    TOKENIZER_PATH,
    DATASET_TO_INFERENCE,
)
from utils import get_inference_model, generate_caption
import json
import tensorflow as tf
import argparse


print("\n\nINFERENCE: {}\n\n".format(DATASET_TO_INFERENCE))

# get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(TOKENIZER_PATH)
tokenizer = tokenizer.layers[1]

# get model
model = get_inference_model(MODEL_CONFIG_PATH)

# load model weights
model.load_weights(MODEL_WEIGHT_PATH)

# generate new caption from input image
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument("--path", help="Path to image file.")
image_path = parser.parse_args().path

with open(MODEL_CONFIG_PATH) as json_file:
    model_config = json.load(json_file)

text_caption = generate_caption(
    image_path, model, tokenizer, model_config["SEQ_LENGTH"]
)

print("PREDICTING_WITH_MODEL: ", MODEL_WEIGHT_PATH)
print("PREDICT CAPTION : %s" % (text_caption))
