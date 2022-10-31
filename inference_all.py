from utils import get_inference_model, generate_caption
import json
import os
import tensorflow as tf
from settings_inference import (
    DATE_TO_EVALUATE,
    TOKENIZER_PATH,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHT_PATH,
)
from itertools import islice

print("\n\nINFERENCE ALL: {}\n\n".format(DATE_TO_EVALUATE))

# get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(TOKENIZER_PATH)
tokenizer = tokenizer.layers[1]

# get model
model = get_inference_model(MODEL_CONFIG_PATH)

# load model weights
model.load_weights(MODEL_WEIGHT_PATH)

list = []

with open(MODEL_CONFIG_PATH) as json_file:
    model_config = json.load(json_file)

with open("datasets/karpathy_val2014_indo.json") as karpathy_val2014_indo:
    data = json.load(karpathy_val2014_indo)

# looping through karpahy test dataset
iteration = 1
for key, value in islice(data.items(), 5000, len(data)):
    dict = {}
    image_path = key
    image_id = (
        key.replace("datasets/val2014/COCO_val2014_", "")
        .replace(".jpg", "")
        .lstrip("0")
    )
    dict["image_id"] = int(image_id)
    dict["caption"] = generate_caption(
        image_path, model, tokenizer, model_config["SEQ_LENGTH"]
    )
    print("Iteration: {}".format(iteration), "DICT: ", dict)
    iteration += 1
    list.append(dict)

# create new directory for saving model
SAVE_DIR = "save_captions/" + DATE_TO_EVALUATE
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


with open("{}/captions_karpathy_test_results_indo.json".format(SAVE_DIR), "w") as fp:
    json.dump(list, fp)

# save config inference
config_inference_all = {
    "MODEL_CONFIG_PATH": MODEL_CONFIG_PATH,
    "MODEL_WEIGHT_PATH": MODEL_WEIGHT_PATH,
}

json.dump(config_inference_all, open(SAVE_DIR + "/config_inference_all.json", "w"))
