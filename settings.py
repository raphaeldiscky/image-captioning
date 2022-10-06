from datetime import datetime

DATE_NOW = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

# CNN Model
CCN_MODEL = "imagenet"
# Desired image dimensions
IMAGE_SIZE = (299, 299)
# Max vocabulary size
MAX_VOCAB_SIZE = 15000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512
# Number of self-attention heads
NUM_HEADS = 6
# Per-layer units in the feed-forward network
FF_DIM = 1024
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 512
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 1


#### COCO dataset
# Max number train dataset images : 82783
# Max number valid dataset images : 40504
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 82783]
NUM_TRAIN_IMG = 1000
# Number of valid images -> it must be a value between [1, 40504]
# N.B. -> IMPORTANT : the number of images of the test set is given by the difference between 40504 and NUM_VALID_IMG values.
# for instance, with NUM_VALID_IMG = 20000 -> valid set have 20000 images and test set have the last 13432 images.
NUM_VALID_IMG = 50
# Data augumention on train set
TRAIN_SET_AUG = True
# Data augmention on valid set
VALID_SET_AUG = False
# If you want to calculate the performance on the test set.
TEST_SET = True

# train_data_json_path = "COCO_dataset/captions_mapping_train_english.json"
# valid_data_json_path = "COCO_dataset/captions_mapping_valid_english.json"
# text_data_json_path = "COCO_dataset/text_data_english.json"

train_data_json_path = "COCO_dataset/mapped_captions_train2014.json"
valid_data_json_path = "COCO_dataset/mapped_captions_val2014.json"
text_data_json_path = "COCO_dataset/text_data_indo.json"

# Save training files directory
SAVE_DIR = "save_train_dir/"
