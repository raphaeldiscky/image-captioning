from matplotlib import pyplot as plt
from settings_inference import DATE_TO_EVALUATE
import json
import os

with open("./save_train_dir/{}/history.json".format(DATE_TO_EVALUATE)) as json_file:
    history_dict = json.load(json_file)


def plot_accuracy():
    # create new directory for saving plot
    SAVE_DIR = "save_plot/" + DATE_TO_EVALUATE
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
    plt.title("Model Accuracy: {}".format(DATE_TO_EVALUATE))
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.xticks(range(1, len(history_dict["acc"]) + 1))
    plt.savefig("./save_plot/{}/model_accuracy.png".format(DATE_TO_EVALUATE))


def plot_loss():
    plt.figure()
    # create new directory for saving plot
    SAVE_DIR = "save_plot/" + DATE_TO_EVALUATE
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
    plt.title("Model Loss: {}".format(DATE_TO_EVALUATE))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.xticks(range(1, len(history_dict["loss"]) + 1))
    plt.savefig("./save_plot/{}/model_loss.png".format(DATE_TO_EVALUATE))


plot_accuracy()
plot_loss()
