
from settings_evaluation import (
    INFERENCE_TEST_RESULT_PATH,
    VAL_RAW_PATH,
    DATE_TO_EVALUATE,
)
import json
import os
import shutil


print("\n\nEVALUATE: {}\n\n".format(DATE_TO_EVALUATE))

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

results = []

# create coco object and coco_result object
coco = COCO(VAL_RAW_PATH)
coco_result = coco.loadRes(INFERENCE_TEST_RESULT_PATH)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# get ids of images
coco_eval.params["image_id"] = coco_result.getImgIds()

# evaluate results
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    results.append(f"{metric}: {score:.3f}")

SAVE_DIR = "save_evaluations/" + DATE_TO_EVALUATE

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# copy training config and result to save_evaluations
config_train_path = "save_trains/" + DATE_TO_EVALUATE + "/config_train.json"
with open(config_train_path) as json_file:
    config_train = json.load(json_file)
target = SAVE_DIR + "/training_config.json"
shutil.copyfile(config_train_path, target)

# copy training history to save_evaluations
history_path = "save_trains/" + DATE_TO_EVALUATE + "/history.json"
with open(history_path) as json_file:
    history = json.load(json_file)
target = SAVE_DIR + "/history.json"
shutil.copyfile(history_path, target)

results.append(
    {
        "EPOCH": len(history["acc"]),
        "CNN_MODEL": config_train["CNN_MODEL"],
        "NUM_HEADS": config_train["NUM_HEADS"],
        "EMBED_DIM": config_train["EMBED_DIM"],
    }
)

json.dump(results, open(SAVE_DIR + "/evaluation_results.json", "w"))
