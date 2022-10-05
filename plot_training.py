from matplotlib import pyplot as plt


def plot_training(history):
    plt.plot(history.history["acc"], label="training accuracy")
    plt.plot(history.history["val_acc"], label="val accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Model Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
