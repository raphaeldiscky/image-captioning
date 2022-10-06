from matplotlib import pyplot as plt
from settings_inference import DATE_TO_EVALUATE
import json

with open("./save_train_dir/{}/history.json".format(DATE_TO_EVALUATE)) as json_file:
    history_dict = json.load(json_file)


def plot_training():
    plt.plot(history_dict["acc"], label="training accuracy")
    plt.plot(history_dict["val_acc"], label="val accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(history_dict["loss"], label="training loss")
    plt.plot(history_dict["val_loss"], label="val loss")
    plt.title("Model Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


plot_training()
