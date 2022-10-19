from datetime import datetime

DATE_NOW = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# CNN Model (efficientnet/resnet)
CCN_MODEL = "efficientnet"
# Desired image dimensions
IMAGE_SIZE = (256, 256)
# Max vocabulary size
MAX_VOCAB_SIZE = 20000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Number of self-attention heads
NUM_HEADS = 8
# Per-layer units in the feed-forward network
FF_DIM = 2048
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 10


# Use karpathy's split of dataset into train, val, test
# Train = 113287
# Val = 5000
# Test = 5000 (remainder)
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 113287]
NUM_TRAIN_IMG = 113287
# Number of valid images -> it must be a value between [1, 10000],
# if NUM_VALID_IMG = 5000, then NUM_TEST_IMG = 5000
NUM_VALID_IMG = 5000
# Data augumention on train set
TRAIN_SET_AUG = True
# Data augmention on valid set
VALID_SET_AUG = False

# For Indonesian dataset
train_data_json_path = "datasets/karpathy_train2014_indo.json"
valid_data_json_path = "datasets/karpathy_val2014_indo.json"
text_data_json_path = "datasets/text_data_indo.json"

# Save training files directory
SAVE_DIR = "save_train_dir/"

## For ENGLISH dataset
# train_data_json_path = "datasets/captions_mapping_train_english.json"
# valid_data_json_path = "datasets/captions_mapping_valid_english.json"
# text_data_json_path = "datasets/text_data_english.json"
