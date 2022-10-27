import matplotlib.pyplot as plt
from IPython import display


def plot_loss_and_accuracy(
    loss_train, loss_val, train_accuracy, val_accuracy, plot_dir, clear_output=True
):
    # if clear_output:
    #     display.clear_output(wait=True)
    _, ax = plt.subplots(1, 2, figsize=(16, 6))

    if len(loss_train) > 0:
        ax[0].plot(loss_train, "*-b", label="train")
        ax[0].plot(loss_val, "*-r", label="test")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[0].set_xlabel("# epochs processed")
        ax[0].set_ylabel("loss value")

    if len(train_accuracy) > 0:
        ax[1].plot(train_accuracy, "*-b", label="train")
        ax[1].plot(val_accuracy, "*-r", label="test")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        ax[1].set_xlabel("# epochs processed")
        ax[1].set_ylabel("accuracy value")

    plt.savefig(f"{plot_dir}/metrics.png")
    # plt.show()


def show_slices(image, axis1="x", axis2="y", axis3="z"):
    slice_0 = image[20, :, :]
    slice_1 = image[:, 20, :]
    slice_2 = image[:, :, 20]
    image = [slice_0, slice_1, slice_2]
    _, axes = plt.subplots(1, len(image), figsize=[15, 15])
    for i, slice in enumerate(image):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        axes[0].set(xlabel=axis2, ylabel=axis3)
        axes[1].set(xlabel=axis1, ylabel=axis3)
        axes[2].set(xlabel=axis1, ylabel=axis2)
    plt.show()
