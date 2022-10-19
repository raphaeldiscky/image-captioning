DATE_TO_EVALUATE = "19_10_2022_19_26_19"  # change this date

# Settings for coco_evaluation.py
INFERENCE_ALL_RESULT_PATH = (
    "./save_captions/{}/captions_val2014_results_indo.json".format(DATE_TO_EVALUATE)
)

VAL_RAW_PATH = (
    "./datasets/captions/captions_raw_val2014_indo.json"  # change this to suitable data
)
TEST_RAW_PATH = "./datasets/captions/captions_raw_test2014_idno.json"  # change this to suitable data
