from datetime import datetime

DATE_NOW = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# CNN model choose between "efficientnet" or "resnet"
CNN_MODEL = "efficientnet"
# early stopping
EARLY_STOPPING = False
# desired image dimensions
IMAGE_SIZE = (256, 256)
# max vocabulary size
MAX_VOCAB_SIZE = 20000
# fixed length allowed for any sequence
SEQ_LENGTH = 25
# dimension for the image embeddings and token embeddings
EMBED_DIM = 1024
# number of self-attention heads
NUM_HEADS = 6
# per-layer units in the feed-forward network
FF_DIM = 2048
# shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# batch size
BATCH_SIZE = 64
# numbers of training epochs
EPOCHS = 12
# dimesion of the linearly projected queries and keys
KEY_DIM = 64
# dimension of the linearly projected values
VALUE_DIM = 64


# USING KARPATHY SPLIT
# train = 113287
# val = 5000
# test = 5000 (remainder)
REDUCE_DATASET = False
# number of train images -> it must be a value between [1, 113287]
NUM_TRAIN_IMG = 113287
# number of valid images -> it must be a value between [1, 10000]
# if NUM_VALID_IMG = 5000, then NUM_TEST_IMG = 5000 is the remainder
NUM_VALID_IMG = 5000
# data augumention on train set
TRAIN_SET_AUG = True
# data augmention on valid set
VALID_SET_AUG = False

# for Indonesian dataset
train_data_json_path = "datasets/karpathy_train2014_indo.json"  # 113287 data
valid_data_json_path = "datasets/karpathy_valtest2014_indo.json"  # 10000 data
captions_data_json_path = "datasets/captions_data_indo.json"  # list of captions

# save training files directory
SAVE_DIR = "save_trains/"
