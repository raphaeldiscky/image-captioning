DATE_TO_EVALUATE = "06_10_2022_23_47_18"  # change this date

# Settings for coco_evaluation.py
INFERENCE_ALL_RESULT_PATH = (
    "./save_captions/{}/captions_val2014_results_indo.json".format(DATE_TO_EVALUATE)
)

VAL_RAW_PATH = (
    "./COCO_dataset/captions/captions_val2014.json"  # change this to suitable data
)
TEST_RAW_PATH = (
    "./COCO_dataset/captions/captions_test2014.json"  # change this to suitable data
)
