# change this date to the date you want to evaluate
DATE_TO_EVALUATE = "23_10_2022_01_31_59"

# For inference.py and inference_all.py
# tokenizer model saved path
TOKENIZER_PATH = "save_train_dir/{}/tokenizer".format(DATE_TO_EVALUATE)
# config model saved path
MODEL_CONFIG_PATH = "save_train_dir/{}/config_train.json".format(DATE_TO_EVALUATE)
# model weight path
MODEL_WEIGHT_PATH = "save_train_dir/{}/model_weights_coco.h5".format(DATE_TO_EVALUATE)


# For inference_all.py
RAW_VAL_IMAGES_DIR = "./datasets/val2014"
RAW_TEST_IMAGES_DIR = "./datasets/test2014"
DATA_TYPE = "val2014"  # change this to "test2014" or "val2014"
LIMIT_LENGTH_DATA = 5000
