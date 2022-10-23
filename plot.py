from matplotlib import pyplot as plt
from settings_inference import DATE_TO_EVALUATE
import numpy as np
import json

with open("./save_train_dir/{}/history.json".format(DATE_TO_EVALUATE)) as json_file:
    history_dict = json.load(json_file)


def plot_training():
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
    plt.show()

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
    plt.show()


plot_training()
