from utils import get_inference_model, generate_caption
import json
import os
import tensorflow as tf
from settings_inference import (
    DATE_TO_INFERENCE,
    TOKENIZER_PATH,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHT_PATH,
    DATASET_TO_INFERENCE,
    SAVE_DIR,
)
from itertools import islice

print("\n\nINFERENCE ALL: {}\n\n".format(DATE_TO_INFERENCE))

# get tokenizer layer from local
tokenizer = tf.keras.models.load_model(TOKENIZER_PATH)
tokenizer = tokenizer.layers[1]


# get model
model = get_inference_model(MODEL_CONFIG_PATH)

# load model weights
model.load_weights(MODEL_WEIGHT_PATH)

list = []

with open(MODEL_CONFIG_PATH) as json_file:
    model_config = json.load(json_file)

with open("datasets/karpathy_valtest2014_indo.json") as karpathy_valtest2014_indo:
    data = json.load(karpathy_valtest2014_indo)

# looping through val or test dataset
iteration = 1
if DATASET_TO_INFERENCE == "test":
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
        print("Iteration: {}".format(iteration), dict)
        iteration += 1
        list.append(dict)
elif DATASET_TO_INFERENCE == "val":
    for key, value in islice(data.items(), 0, 5000):
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
        print("Iteration: {}".format(iteration), dict)
        iteration += 1
        list.append(dict)

# create new directory for saving model
SAVE_DIR = "save_captions/" + DATE_TO_INFERENCE
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


with open(
    "{}/captions_karpathy_{}_results_indo.json".format(SAVE_DIR, DATASET_TO_INFERENCE),
    "w",
) as fp:
    json.dump(list, fp)

# save config inference
config_inference_all = {
    "INFERENCED_DATASET": DATASET_TO_INFERENCE,
    "SELECTED_MODEL": DATE_TO_INFERENCE,
}

json.dump(config_inference_all, open(SAVE_DIR + "/config_inference_all.json", "w"))
