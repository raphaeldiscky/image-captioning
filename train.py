import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from settings_train import (
    BATCH_SIZE,
    CNN_MODEL,
    DATE_NOW,
    EMBED_DIM,
    EPOCHS,
    FF_DIM,
    IMAGE_SIZE,
    NUM_HEADS,
    NUM_TRAIN_IMG,
    NUM_VALID_IMG,
    SAVE_DIR,
    SHUFFLE_DIM,
    VALID_SET_AUG,
    train_data_json_path,
    valid_data_json_path,
    text_data_json_path,
    REDUCE_DATASET,
    MAX_VOCAB_SIZE,
    SEQ_LENGTH,
    TRAIN_SET_AUG,
    EARLY_STOPPING,
)
from datasets import (
    make_dataset,
    custom_standardization,
    reduce_dataset_dim,
    valid_test_split,
)
from custom_schedule import custom_schedule
from models import (
    get_cnn_model,
    Decoder,
    Encoder,
    ImageCaptioningModel,
)
from utils import save_tokenizer
import json
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# load dataset
with open(train_data_json_path) as json_file:
    train_data = json.load(json_file)
with open(valid_data_json_path) as json_file:
    valid_data = json.load(json_file)
with open(text_data_json_path) as json_file:
    text_data = json.load(json_file)

# for reduce number of images in the dataset
if REDUCE_DATASET:
    train_data, valid_data = reduce_dataset_dim(train_data, valid_data)

print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

# define tokeziner / vectorize layer
tokenizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

# adapt tokenizer to create the vocabulary
tokenizer.adapt(text_data)

# define vocabulary size of the vocabulary
VOCAB_SIZE = len(tokenizer.get_vocabulary())

print("Vocab size: ", VOCAB_SIZE)

# split dataset to valid and test set
valid_data, test_data = valid_test_split(valid_data)

print("Validation data after splitting with test set: ", len(valid_data))
print("Test data: ", len(test_data))

config_train = {
    "CNN_MODEL": CNN_MODEL,
    "EARLY_STOPPING": EARLY_STOPPING,
    "IMAGE_SIZE": IMAGE_SIZE,
    "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
    "SEQ_LENGTH": SEQ_LENGTH,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "FF_DIM": FF_DIM,
    "SHUFFLE_DIM": SHUFFLE_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "VOCAB_SIZE": VOCAB_SIZE,
    "NUM_TRAIN_IMG": NUM_TRAIN_IMG,
    "NUM_VALID_IMG": NUM_VALID_IMG,
    "NUM_TEST_IMG": len(test_data),
}

print(config_train)

# setting batch dataset
train_dataset = make_dataset(
    list(train_data.keys()),  # path to images
    list(train_data.values()),  # captions
    data_aug=TRAIN_SET_AUG,
    tokenizer=tokenizer,
)

valid_dataset = make_dataset(
    list(valid_data.keys()),
    list(valid_data.values()),
    data_aug=VALID_SET_AUG,
    tokenizer=tokenizer,
)

test_dataset = make_dataset(
    list(test_data.keys()),
    list(test_data.values()),
    data_aug=False,
    tokenizer=tokenizer,
)

# get model
cnn_model = get_cnn_model(CNN_MODEL)

encoder = Encoder(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)

decoder = Decoder(
    embed_dim=EMBED_DIM,
    ff_dim=FF_DIM,
    num_heads=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
)

caption_model = ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder
)

# define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

# early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

# create a learning rate schedule
lr_scheduler = custom_schedule(EMBED_DIM)
optimizer = keras.optimizers.Adam(
    learning_rate=lr_scheduler, beta_1=0.9, beta_2=0.98, epsilon=1e-9
)

# compile the model
caption_model.compile(optimizer=optimizer, loss=cross_entropy)

# fit the model
history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping] if EARLY_STOPPING else None,
)


# compute definitive metrics on train/valid/test set
train_metrics = caption_model.evaluate(train_dataset, batch_size=BATCH_SIZE)
valid_metrics = caption_model.evaluate(valid_dataset, batch_size=BATCH_SIZE)
test_metrics = caption_model.evaluate(test_dataset, batch_size=BATCH_SIZE)

# create new directory for saving model
NEW_DIR = SAVE_DIR + DATE_NOW
os.mkdir(NEW_DIR)

# save training history under the form of a json file
history_dict = history.history
json.dump(history_dict, open(SAVE_DIR + "{}/history.json".format(DATE_NOW), "w"))


# save weights model
caption_model.save_weights(SAVE_DIR + "{}/model_weights_coco.h5".format(DATE_NOW))

# save metrics results
metrics_results = {
    "TRAIN_SET": "Train Loss = %.4f - Train Accuracy = %.4f"
    % (train_metrics[0], train_metrics[1]),
    "VALID_SET": "Valid Loss = %.4f - Valid Accuracy = %.4f"
    % (valid_metrics[0], valid_metrics[1]),
    "TEST_SET": "Test Loss = %.4f - Test Accuracy = %.4f"
    % (test_metrics[0], test_metrics[1]),
}
print(metrics_results)

json.dump(
    metrics_results, open(SAVE_DIR + "{}/metrics_results.json".format(DATE_NOW), "w")
)

# save config model train
json.dump(config_train, open(SAVE_DIR + "{}/config_train.json".format(DATE_NOW), "w"))

# save tokenizer model
save_tokenizer(tokenizer, NEW_DIR)
