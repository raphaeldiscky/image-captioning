from utils import get_inference_model, generate_caption
import json
import tensorflow as tf
import os
from settings_inference import (
    DATE_TO_EVALUATE,
    TOKENIZER_PATH,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHT_PATH,
    RAW_VAL_IMAGES_DIR,
    RAW_TEST_IMAGES_DIR,
    LIMIT_LENGTH_DATA,
    DATA_TYPE,
)

SELECTED_RAW_IMAGES_DIR = (
    RAW_VAL_IMAGES_DIR
    if DATA_TYPE == "val2014"
    else RAW_TEST_IMAGES_DIR
    if DATA_TYPE == "test2014"
    else None
)

print("INFERENCE: {}".format(DATE_TO_EVALUATE))

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

# looping through images in the selected directory
iteration = 1
for filename in os.listdir(SELECTED_RAW_IMAGES_DIR)[:LIMIT_LENGTH_DATA]:
    dict = {}
    image_id = (
        filename.replace("COCO_{}_".format(DATA_TYPE), "")
        .replace(".jpg", "")
        .lstrip("0")
    )
    image_path = os.path.join(SELECTED_RAW_IMAGES_DIR, filename).replace("\\", "/")
    dict["image_id"] = int(image_id)
    dict["caption"] = generate_caption(
        image_path, model, tokenizer, model_config["SEQ_LENGTH"]
    )
    print("Iteration: {}, FILENAME: ".format(iteration), filename, "DICT: ", dict)
    iteration += 1
    list.append(dict)

# create new directory for saving model
SAVE_DIR = "save_captions/" + DATE_TO_EVALUATE
os.mkdir(SAVE_DIR)


with open("{}/captions_{}_results_indo.json".format(SAVE_DIR, DATA_TYPE), "w") as fp:
    json.dump(list, fp)

# save config inference
config_inference_all = {
    "MODEL_CONFIG_PATH": MODEL_CONFIG_PATH,
    "MODEL_WEIGHT_PATH": MODEL_WEIGHT_PATH,
    "DATA_TYPE": DATA_TYPE,
    "LIMIT_LENGTH_DATA": LIMIT_LENGTH_DATA,
}


json.dump(config_inference_all, open(SAVE_DIR + "/config_inference_all.json", "w"))
