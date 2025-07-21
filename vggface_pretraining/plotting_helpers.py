import matplotlib.pyplot as plt
import torch

def show_triplets(dataloader, n=5):
    """
    Displays `n` triplets (anchor, positive, negative) from a triplet DataLoader.

    Args:
        dataloader (DataLoader): DataLoader that yields (anchor, positive, negative) triplets.
        n (int): Number of triplets to display.
    """
    data_iter = iter(dataloader)
    shown = 0

    plt.figure(figsize=(12, 4 * n))

    while shown < n:
        try:
            anchors, positives, negatives = next(data_iter)
        except StopIteration:
            print("End of DataLoader reached.")
            break

        for i in range(anchors.size(0)):
            if shown >= n:
                break

            for j, img in enumerate([anchors[i], positives[i], negatives[i]]):
                img = img.squeeze(0) if img.shape[0] == 1 else img  # Remove channel dim if grayscale
                plt.subplot(n, 3, shown * 3 + j + 1)
                plt.imshow(img.numpy(), cmap='gray')
                if j == 0:
                    plt.title("Anchor")
                elif j == 1:
                    plt.title("Positive")
                else:
                    plt.title("Negative")
                plt.axis('off')

            shown += 1

    plt.tight_layout()
    plt.show()

def plot_loss_curve(train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss', marker='o')
    plt.plot(val_loss, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_accuracy_curve(train_acc, val_acc):
    """
    Plots the training and validation accuracy curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='Train Accuracy', marker='o')
    plt.plot(val_acc, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Triplet Embedding Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()