# change this date to the date you want to inference and plot
DATE_TO_EVALUATE = "21_10_2022_01_12_27"

# For inference.py and inference_all.py
# tokenizer model saved path
TOKENIZER_PATH = "save_trains/{}/tokenizer".format(DATE_TO_EVALUATE)
# config model saved path
MODEL_CONFIG_PATH = "save_trains/{}/config_train.json".format(DATE_TO_EVALUATE)
# model weight path
MODEL_WEIGHT_PATH = "save_trains/{}/model_weights_coco.h5".format(DATE_TO_EVALUATE)
