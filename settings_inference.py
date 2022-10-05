# @Settings for generate json and inference
DATE_TO_EVALUATE = "04_10_2022_23_16_28"  # change this date

# Tokenizer model saved path
TOKENIZER_PATH = "save_train_dir/{}/tokenizer".format(DATE_TO_EVALUATE)
# Config model saved path
MODEL_CONFIG_PATH = "save_train_dir/{}/config_train.json".format(DATE_TO_EVALUATE)
# Model weight path
MODEL_WEIGHT_PATH = "save_train_dir/{}/model_weights_coco.h5".format(DATE_TO_EVALUATE)

PATH_VAL_DIR = "./COCO_dataset/val2014"
PATH_TEST_DIR = "./COCO_dataset/test2014"
TOTAL_DATA = 5000
