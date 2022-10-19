DATE_TO_EVALUATE = "19_10_2022_19_26_19"  # change this date

# For inference.py and inference_all.py
# Tokenizer model saved path
TOKENIZER_PATH = "save_train_dir/{}/tokenizer".format(DATE_TO_EVALUATE)
# Config model saved path
MODEL_CONFIG_PATH = "save_train_dir/{}/config_train.json".format(DATE_TO_EVALUATE)
# Model weight path
MODEL_WEIGHT_PATH = "save_train_dir/{}/model_weights_coco.h5".format(DATE_TO_EVALUATE)


# For inference_all.py
RAW_VAL_IMAGES_DIR = "./datasets/val2014"
RAW_TEST_IMAGES_DIR = "./datasets/test2014"
DATA_TYPE = "val2014"  # change this to "test2014" for testing or "val2014"
LIMIT_LENGTH_DATA = 5000
