import os
import torch
import kagglehub
from data import FERDataModule
from models import EmotionDetector
from training import Trainer, Test, get_criterion
from utils import Utils, test_dataloader_shapes, plot_samples_from_dataloader
from config import Config
import torch.nn as nn

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
config = Config()

# Set random seed
Utils.set_seed(config.SEED)

# Init data module
data_module = FERDataModule(csv_path=os.path.join(config.DATASET_PATH, 'fer2013.csv'),
                            batch_size=config.BATCH_SIZE,
                            image_size=config.IMAGE_SIZE[0],
                            criterion=config.CRITERION)
data_module.setup()

# Create dataloaders
train_loader = data_module.train_dataloader(num_classes=config.NUM_CLASSES)
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Initialize model
model = EmotionDetector(num_classes=config.NUM_CLASSES).to(device)

criterion = get_criterion(config, data_module.get_class_weights())

# Initialize trainer
trainer = Trainer(model, train_loader, val_loader, config, device, criterion)
history, best_epoch, best_f1, best_acc, best_model = trainer.train()

# Plot training performance
Utils.plot_loss_curve(history['train_loss'], history['val_loss'], show=True, save=True)
Utils.plot_accuracy_curve(history['train_acc'], history['val_acc'], show=True, save=True)
Utils.plot_f1_score(history['val_f1'], show=True, save=True)

tester = Test(best_model, test_loader, criterion, config)


# Run test and get predictions
test_loss, test_acc, precision, recall, f1, all_preds, all_labels = tester.test()

# Plot confusion matrix
Utils.plot_confusion_matrix(all_labels, all_preds, class_names=data_module.get_class_names())
