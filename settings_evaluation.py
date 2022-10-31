# change this date to the date you want to evaluate
DATE_TO_EVALUATE = "21_10_2022_01_12_27"

INFERENCE_ALL_RESULT_PATH = (
    "./save_captions/{}/captions_karpathy_test_results_indo.json".format(
        DATE_TO_EVALUATE
    )
)

VAL_RAW_PATH = "./datasets/captions/captions_raw_val2014_indo.json"
TEST_RAW_PATH = "./datasets/captions/captions_raw_test2014_indo.json"
