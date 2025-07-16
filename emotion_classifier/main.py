import torch

from data_handling.ExpWDataModule import *
from models.model_emotion import EmotionDetector
from training.training_loop import train_model, test_model
from visualization.Visualization import *


def main():
    # Config
    label_file='/kaggle/input/expwds/label/label.lst'
    img_dir='/kaggle/input/expwds/origin'
    batch_size = 128
    num_classes = 7
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "best_emotion_model.pth"
    fine_tune_mode = "partial"  # "none", "partial", or "full"

    # Prepare data
    data_module = ExpWDataModule(label_file, img_dir, batch_size, r_s = 42)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Initialize model
    model = EmotionDetector(num_classes=num_classes, fine_tune=fine_tune_mode)

    # Access original dataset for loss calculation
    original_train_dataset = data_module.train_dataset.dataset

    # Train
    history = train_model(model, train_loader, val_loader, num_epochs, device, save_path,
                          train_dataset=original_train_dataset)




    original_test_dataset = data_module.test_dataset.dataset
    # Test
    test_loss, test_acc, precision, recall, f1, all_preds, all_labels = test_model(model, test_loader, device, test_dataset=original_test_dataset)

    # Visualizations
    plot_loss_curve(history['train_loss'], history['val_loss'])
    plot_accuracy_curve(history['train_acc'], history['val_acc'])
    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    main()
