from settings_inference import DATE_TO_EVALUATE

# Settings for coco_evaluation.py
RESULT_CAPTIONS_PATH = "./save_captions/{}/captions_val2014_results.json".format(
    DATE_TO_EVALUATE
)

VAL_ANNOTATION_PATH = "./COCO_dataset/captions/captions_val2014_indo.json"
TEST_ANNOTATION_PATH = "./COCO_dataset/captions/captions_test2014.json"
