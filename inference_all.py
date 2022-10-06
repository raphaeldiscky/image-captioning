from utility import get_inference_model, generate_caption
import json
import tensorflow as tf
import os
from settings_inference import (
    DATE_TO_EVALUATE,
    TOKENIZER_PATH,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHT_PATH,
    PATH_VAL_DIR,
    TOTAL_DATA,
)

# Get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(TOKENIZER_PATH)
tokenizer = tokenizer.layers[1]

# Get model
model = get_inference_model(MODEL_CONFIG_PATH)

# Load model weights
model.load_weights(MODEL_WEIGHT_PATH)

list = []

with open(MODEL_CONFIG_PATH) as json_file:
    model_config = json.load(json_file)

for filename in os.listdir(PATH_VAL_DIR)[:TOTAL_DATA]:
    dict = {}
    image_id = filename.replace("COCO_val2014_", "").replace(".jpg", "").lstrip("0")
    image_path = os.path.join(PATH_VAL_DIR, filename).replace("\\", "/")
    dict["image_id"] = int(image_id)
    dict["caption"] = generate_caption(
        image_path, model, tokenizer, model_config["SEQ_LENGTH"]
    )
    print("FILENAME: ", filename, "DICT: ", dict)
    list.append(dict)

# Create new directory for saving model
NEW_DIR = "save_captions/" + DATE_TO_EVALUATE
os.mkdir(NEW_DIR)

with open("{}/captions_val2014_results_indo.json".format(NEW_DIR), "w") as fp:
    json.dump(list, fp)
