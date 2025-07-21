import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import random
import numpy as np
import os

from emotion_classifier.config import Config


class Utils:
    @staticmethod
    def _maybe_save(fig, filename, save):
        if save:
            os.makedirs(Config().CHECKPOINT_PATH, exist_ok=True)
            fig.savefig(os.path.join(Config().CHECKPOINT_PATH, filename))
        plt.close(fig)

    @staticmethod
    def plot_loss_curve(train_loss, val_loss, show=True, save=True):
        """
        Plot training and validation loss curves.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'loss_curve.png', save)

    @staticmethod
    def plot_accuracy_curve(train_acc, val_acc, show=True, save=True):
        """
        Plot training and validation accuracy curves.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(train_acc, label='Train Accuracy', marker='o')
        plt.plot(val_acc, label='Validation Accuracy', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'accuracy_curve.png', save)

    @staticmethod
    def plot_f1_score(f1_scores, show=True, save=True):
        """
        Plot validation F1 score curve.
        """
        fig = plt.figure(figsize=(10, 5))
        plt.plot(f1_scores, label='Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Curve')
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()
        Utils._maybe_save(fig, 'f1_score_curve.png', save)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names, show=True, save=True):
        """
        Plot confusion matrix and print classification report.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if show:
            plt.show()
        Utils._maybe_save(fig, 'confusion_matrix.png', save)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    @staticmethod
    def set_seed(seed=None, seed_torch=True):
        """
        Set random seed for reproducibility.
        """
        if seed is None:
            seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
        if seed_torch:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        print(f'Random seed {seed} has been set for reproducibility.')
