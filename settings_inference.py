# change this date to the date you want to inference and plot
DATE_TO_INFERENCE = "11_12_2022_21_32_16"

# choose to validate on "test" or "val" dataset
DATASET_TO_INFERENCE = "test"
# tokenizer model saved path
TOKENIZER_PATH = "save_trains/{}/tokenizer".format(DATE_TO_INFERENCE)
# config model saved path
MODEL_CONFIG_PATH = "save_trains/{}/config_train.json".format(DATE_TO_INFERENCE)
# model weight path
MODEL_WEIGHT_PATH = "save_trains/{}/model_weights_coco.h5".format(DATE_TO_INFERENCE)

SAVE_DIR = "save_captions/"
