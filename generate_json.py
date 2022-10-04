from utility import get_inference_model, generate_caption
import json
import tensorflow as tf
from settings_inference import *
import os
import json
from settings import PATH_VAL_DIR, TOTAL_DATA

# Get tokenizer layer from disk
tokenizer = tf.keras.models.load_model(tokernizer_path)
tokenizer = tokenizer.layers[1]

# Get model
model = get_inference_model(get_model_config_path)

# Load model weights
model.load_weights(get_model_weights_path)

list = []

with open(get_model_config_path) as json_file:
    model_config = json.load(json_file)

for filename in os.listdir(PATH_VAL_DIR)[:TOTAL_DATA]:
    dict = {}
    image_id = filename.replace("COCO_val2014_", "").replace(".jpg", "").lstrip("0")
    image_path = os.path.join(PATH_VAL_DIR, filename).replace("\\", "/")
    dict["image_id"] = int(image_id)
    dict["caption"] = generate_caption(
        image_path, model, tokenizer, model_config["SEQ_LENGTH"]
    )
    print(filename, dict)
    list.append(dict)

with open("captions_val2014_results.json", "w") as fp:
    json.dump(list, fp)
