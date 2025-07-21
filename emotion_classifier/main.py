import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from emotion_classifier.config import Config
from emotion_classifier.data import FERDataModule
from emotion_classifier.training.losses import get_criterion
from emotion_classifier.training.tester import Test
from emotion_classifier.training.trainer import Trainer
from emotion_classifier.utils.plotting import Utils
from vggface_pretraining.models import TripletClassifier

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
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

def initialize_optimizer(model, base_lr=1e-5, mid_lr=5e-5, head_lr=1e-3, weight_decay=1e-4):
    """
    Custom optimizer setup for TripletClassifier:
    - Backbone early layers: base_lr
    - Deeper layers (e.g., layer3/layer4): mid_lr
    - Classifier head: head_lr
    """
    param_groups = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            param_groups.append({'params': param, 'lr': head_lr})
        elif 'layer4' in name or 'layer3' in name:  # deeper resnet blocks
            param_groups.append({'params': param, 'lr': mid_lr})
        elif 'embedding_model' in name or 'features' in name:
            param_groups.append({'params': param, 'lr': base_lr})
        else:
            param_groups.append({'params': param, 'lr': base_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer

# Initialize model
model = TripletClassifier(
    weights_path="/kaggle/input/resnet5-vgg-face/pytorch/v1/1/best_triplet_model.pth",
    num_classes=config.NUM_CLASSES,
    embedding_dim=512,
    freeze_until=4,
    device='cuda',
    model_name="resnet50",
)
progressive_unfreezing_frequency = 4
optimizer = initialize_optimizer(model=model, base_lr=1e-4, mid_lr=5e-4, head_lr=1e-2, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=progressive_unfreezing_frequency*2, gamma=0.1)
criterion = get_criterion(config, data_module.get_class_weights())
# Initialize trainer
trainer = Trainer(model, train_loader, val_loader, config, device, criterion, optimizer=optimizer,scheduler=scheduler, progressive_unfreezing_frequency=progressive_unfreezing_frequency)
history, best_epoch, best_f1, best_acc, best_model = trainer.train()

# Plot training performance
Utils.plot_loss_curve(history['train_loss'], history['val_loss'], show=True, save=True)
Utils.plot_accuracy_curve(history['train_acc'], history['val_acc'], show=True, save=True)
Utils.plot_f1_score(history['val_f1'], show=True, save=True)

# Run test
best_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
best_model = TripletClassifier(
    num_classes=config.NUM_CLASSES,
    embedding_dim=512,
    freeze_until=4,
    device='cuda',
    model_name="resnet50",
    state_dict=best_dict
)
tester = Test(best_model, test_loader, criterion, config)
test_results = tester.test(class_names=data_module.get_class_names())

# Save final model
torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
print(f"✅ Final model saved at: {config.FINAL_MODEL_PATH}")