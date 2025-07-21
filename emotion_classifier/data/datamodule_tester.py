import torch
from datamodule import FERDataModule

def test_dataloader_shapes(dataloader, expected_shape=(1, 128, 128), num_classes=5):
    images, labels = next(iter(dataloader))
    assert isinstance(images, torch.Tensor), "Images are not torch tensors."
    assert isinstance(labels, torch.Tensor), "Labels are not torch tensors."

    assert images.shape[1:] == expected_shape, f"Expected shape {expected_shape}, got {images.shape[1:]}"
    assert labels.max().item() < num_classes, f"Label exceeds expected range {num_classes - 1}"
    print(f"✅ Batch shape: {images.shape}, Labels: {labels.tolist()}")


def test_data_module(dm):
    dm.setup()

    print("\nTesting Train DataLoader:")
    test_dataloader_shapes(dm.train_dataloader())

    print("\nTesting Validation DataLoader:")
    test_dataloader_shapes(dm.val_dataloader())

    print("\nTesting Test DataLoader:")
    test_dataloader_shapes(dm.test_dataloader())

    print("\nClass names:", dm.get_class_names())
    print("Class weights:", dm.get_class_weights())

def plot_samples_from_dataloader(dataloader, class_map, n=16):
    """
    Plot a grid of n samples from the dataloader with their labels.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    images, labels = next(iter(dataloader))
    images = images[:n]
    labels = labels[:n]

    # Unnormalize: (img * std + mean)
    images = images * 0.5 + 0.5

    n_cols = int(np.sqrt(n))
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n):
        img = images[i].squeeze().cpu().numpy()  # shape (128, 128)
        label = labels[i].item()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(class_map.get(label, str(label)))
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    csv_path = "/kaggle/working/fer2013/fer2013.csv"
    dm = FERDataModule(csv_path, batch_size=32, image_size=128)

    # Run tests
    test_data_module(dm)

    # Plot training samples
    print("\n🖼️ Plotting training samples...")
    plot_samples_from_dataloader(dm.train_dataloader(), class_map=dm.emotion_map)

    # Plot validation samples
    print("\n🖼️ Plotting validation samples...")
    plot_samples_from_dataloader(dm.val_dataloader(), class_map=dm.emotion_map)