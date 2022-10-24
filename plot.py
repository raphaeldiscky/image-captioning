from matplotlib import pyplot as plt
from settings_inference import DATE_TO_EVALUATE, MODEL_CONFIG_PATH
import json
import os

with open("./save_trains/{}/history.json".format(DATE_TO_EVALUATE)) as json_file:
    history_dict = json.load(json_file)

with open(MODEL_CONFIG_PATH) as json_file:
    model_config = json.load(json_file)

CNN_MODEL = model_config["CNN_MODEL"]
NUM_HEADS = model_config["NUM_HEADS"]
EMBED_DIM = model_config["EMBED_DIM"]
FF_DIM = model_config["FF_DIM"]


def plot_accuracy():
    # create new directory for saving plot
    SAVE_DIR = "save_plots/" + DATE_TO_EVALUATE
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # plot training & validation accuracy values
    plt.plot(
        range(1, len(history_dict["acc"]) + 1),
        history_dict["acc"],
        label="training accuracy",
    )
    plt.plot(
        range(1, len(history_dict["val_acc"]) + 1),
        history_dict["val_acc"],
        label="validation accuracy",
    )
    plt.title(
        "cnn_model: {}, num_heads: {}, embed_dim: {}, ff_dim: {}".format(
            CNN_MODEL, NUM_HEADS, EMBED_DIM, FF_DIM
        ),
        fontsize="medium",
    )
    plt.suptitle("Model Accuracy", fontsize="large")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.xticks(range(1, len(history_dict["acc"]) + 1))
    plt.savefig("./save_plots/{}/model_accuracy.png".format(DATE_TO_EVALUATE))


def plot_loss():
    plt.figure()
    # create new directory for saving plot
    SAVE_DIR = "save_plots/" + DATE_TO_EVALUATE
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # plot training & validation loss values
    plt.plot(
        range(1, len(history_dict["loss"]) + 1),
        history_dict["loss"],
        label="training loss",
    )
    plt.plot(
        range(1, len(history_dict["val_loss"]) + 1),
        history_dict["val_loss"],
        label="val loss",
    )
    plt.title(
        "cnn_model: {}, num_heads: {}, embed_dim: {}, ff_dim: {}".format(
            CNN_MODEL, NUM_HEADS, EMBED_DIM, FF_DIM
        ),
        fontsize="medium",
    )
    plt.suptitle("Model Loss", fontsize="large")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.xticks(range(1, len(history_dict["loss"]) + 1))
    plt.savefig("./save_plots/{}/model_loss.png".format(DATE_TO_EVALUATE))


plot_accuracy()
plot_loss()
