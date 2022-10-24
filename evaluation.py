from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from settings_evaluation import (
    INFERENCE_ALL_RESULT_PATH,
    VAL_RAW_PATH,
    DATE_TO_EVALUATE,
)
import json
import os
import shutil


print("\n\nEVALUATE: {}\n\n".format(DATE_TO_EVALUATE))

results = []

# create coco object and coco_result object
coco = COCO(VAL_RAW_PATH)
coco_result = coco.loadRes(INFERENCE_ALL_RESULT_PATH)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params["image_id"] = coco_result.getImgIds()

# evaluate results
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.3f}")
    results.append(f"{metric}: {score:.3f}")

SAVE_DIR = "save_evaluations/" + DATE_TO_EVALUATE

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

json.dump(results, open(SAVE_DIR + "/evaluation_results.json", "w"))

# copy training config and result to save_evaluations
original = "save_trains/" + DATE_TO_EVALUATE + "/config_train.json"
target = SAVE_DIR + "/training_config.json"
shutil.copyfile(original, target)

# copy training history to save_evaluations
original = "save_trains/" + DATE_TO_EVALUATE + "/history.json"
target = SAVE_DIR + "/history.json"
shutil.copyfile(original, target)
