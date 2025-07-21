from sklearn.metrics import f1_score
import os
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import copy
import numpy as np
import random
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def compute_accuracy(anchor, positive, negative):
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    correct = (pos_dist < neg_dist).sum().item()
    total = anchor.size(0)
    return correct, total

def train_triplet_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path='best_model.pth', use_tqdm=True, scheduler=None):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n' + '-'*30)

        # -------- Train --------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_preds = []
        train_labels = []

        if use_tqdm:
            train_loader = tqdm(train_loader, desc='training')

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Accuracy and F1
            correct, total = compute_accuracy(anchor_out, positive_out, negative_out)
            train_correct += correct
            train_total += total

            # For F1: prediction = 1 if pos < neg else 0
            preds = (F.pairwise_distance(anchor_out, positive_out) < F.pairwise_distance(anchor_out, negative_out)).long().cpu().numpy()
            labels = np.ones_like(preds)  # True class is always 1 (pos closer)
            train_preds.extend(preds)
            train_labels.extend(labels)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        train_f1 = f1_score(train_labels, train_preds)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['train_f1'].append(train_f1)

        print(f'Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f} | F1: {train_f1:.4f}')

        # -------- Validate --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            if use_tqdm:
                val_loader = tqdm(val_loader, desc='validating')
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                loss = criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()

                # Accuracy and F1
                correct, total = compute_accuracy(anchor_out, positive_out, negative_out)
                val_correct += correct
                val_total += total

                preds = (F.pairwise_distance(anchor_out, positive_out) < F.pairwise_distance(anchor_out, negative_out)).long().cpu().numpy()
                labels = np.ones_like(preds)
                val_preds.extend(preds)
                val_labels.extend(labels)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds)
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['val_f1'].append(val_f1)

        print(f'Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f}')

        # -------- Save Best Model --------
        if avg_val_loss < best_val_loss:
            print(f'✅ Saving best model (Val Loss improved from {best_val_loss:.4f} → {avg_val_loss:.4f})')
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)

    print(f'\n🏁 Training complete. Best Val Loss: {best_val_loss:.4f}')

    # Load and return the best model + training history
    model.load_state_dict(best_model_wts)
    return model, history