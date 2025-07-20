import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, criterion, EARLY_STOP_PATIENCE=None):
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': []
        }
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.early_stop_patience = EARLY_STOP_PATIENCE if EARLY_STOP_PATIENCE is not None else self.config.EARLY_STOP_PATIENCE
        self.criterion = criterion
        
        # Enhanced optimizer with better regularization
        self.optimizer = optim.AdamW(  # AdamW instead of Adam for better regularization
            model.parameters(), 
            lr=self.config.LEARNING_RATE, 
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enhanced learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=7, 
            verbose=True,
            min_lr=getattr(self.config, 'MIN_LR', 1e-7)
        )
        
        # Add gradient clipping
        self.gradient_clipping = getattr(self.config, 'GRADIENT_CLIPPING', None)
        
        # Track overfitting metrics
        self.overfitting_threshold = 15.0  # If train_acc - val_acc > 15%, consider overfitting

    def train(self):
        best_f1 = 0.0
        best_epoch = 0
        best_acc = 0.0
        early_stop_counter = 0
        best_val_loss = float('inf')
        best_model = None

        for epoch in range(self.config.NUM_EPOCHS):
            # Training phase
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc, val_f1, all_preds, all_labels = self._validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Update scheduler
            self.scheduler.step(val_loss)

            # Check for overfitting
            overfitting_gap = train_acc - val_acc
            is_overfitting = overfitting_gap > self.overfitting_threshold

            # Save best model based on F1 score
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch + 1
                best_acc = val_acc
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                best_model = self.model
                early_stop_counter = 0
                print(f"✅ [Epoch {epoch+1}] Best model saved - Val F1: {best_f1:.4f}, Acc: {best_acc:.2f}%")
            else:
                early_stop_counter += 1

            # Enhanced early stopping conditions
            should_stop = False
            
            # Standard early stopping
            if early_stop_counter >= self.early_stop_patience:
                print(f"Early stopping: No improvement for {self.early_stop_patience} epochs")
                should_stop = True
            
            # Overfitting-based early stopping
            if is_overfitting and train_acc > 85 and epoch > 10:
                print(f"Early stopping: Severe overfitting detected (gap: {overfitting_gap:.1f}%)")
                should_stop = True
            
            # Loss explosion check
            if val_loss > 10.0 and epoch > 5:
                print(f"Early stopping: Loss explosion detected (val_loss: {val_loss:.4f})")
                should_stop = True
                
            if should_stop:
                break

            # Enhanced logging with overfitting detection
            overfitting_status = "🔥 OVERFITTING" if is_overfitting else "✅ Normal"
            lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f} | "
                  f"Gap: {overfitting_gap:.1f}% {overfitting_status} | LR: {lr:.2e}")

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"Best Epoch: {best_epoch}")
        print(f"Best Validation F1: {best_f1:.4f}")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print(f"Model saved at: {self.config.BEST_MODEL_PATH}")
        print(f"{'='*60}")

        return self.history, best_epoch, best_f1, best_acc, best_model

    def _train_epoch(self, epoch):
        """Training phase for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]") \
            if self.config.USE_TQDM else self.train_loader

        for inputs, labels in train_bar:
            if isinstance(labels, tuple):  # MixUp/CutMix case
                labels_a, labels_b, lam = labels
                inputs = inputs.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                labels = labels_a  # Use primary label for accuracy
            else:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if isinstance(labels, tuple):
                loss = self.criterion(outputs, labels_a, labels_b, lam)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            
            # Apply gradient clipping if specified
            if self.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
            
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if self.config.USE_TQDM:
                train_bar.set_postfix(
                    loss=running_loss / total, 
                    acc=100. * correct / total,
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def _validate_epoch(self, epoch):
        """Validation phase for one epoch."""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        val_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]") \
            if self.config.USE_TQDM else self.val_loader

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                # Use appropriate loss calculation
                if hasattr(self.criterion, 'forward') and 'lam' in self.criterion.forward.__code__.co_varnames:
                    loss = self.criterion(outputs, labels, lam=1.0)  # Standard CE for validation
                else:
                    loss = self.criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if self.config.USE_TQDM:
                    val_bar.set_postfix(
                        loss=val_loss / val_total, 
                        acc=100. * val_correct / val_total
                    )

        epoch_val_loss = val_loss / len(self.val_loader.dataset)
        epoch_val_acc = 100. * val_correct / val_total

        # Calculate F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        return epoch_val_loss, epoch_val_acc, f1, all_preds, all_labels
