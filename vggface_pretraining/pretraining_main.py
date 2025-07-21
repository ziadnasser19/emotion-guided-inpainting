# Set device
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from vggface_pretraining.dataset import TripletVGGFaceDataset
from vggface_pretraining.models import TripletNetwork
from vggface_pretraining.training import train_triplet_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Instantiate the model
model = TripletNetwork(model_name='resnet50', embedding_dim=512)

# 2. Define the loss function (Triplet Loss)
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# 3. Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, verbose=True)

# 4. Data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
])

train_dataset = TripletVGGFaceDataset(
    root_dir='/kaggle/input/vggface2/train',
    transform=train_transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

val_dataset = TripletVGGFaceDataset(
    root_dir='/kaggle/input/vggface2/train',
    transform=val_transform
)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# 5. Train and save the best model
best_model, history = train_triplet_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=80,
    device=device,
    save_path='best_triplet_model.pth',
    use_tqdm=False,
    scheduler=scheduler
)