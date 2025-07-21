from sklearn.metrics import precision_recall_fscore_support
import torch
from tqdm import tqdm
from .losses import *
from ..utils.plotting import Utils


class Test:
    def __init__(self, model, test_loader, criterion, config):
        """
        Initialize the Test class for model evaluation.
        
        Args:
            model (nn.Module): Trained model.
            test_loader (DataLoader): Test DataLoader.
            criterion (nn.Module): Loss function.
            config (Config): Configuration object.
        """
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test(self, xai_interpreter=None, class_names=None):
        """
        Evaluate the model on the test dataset and generate XAI heatmaps.
        
        Args:
            xai_interpreter (XAIInterpreter, optional): XAI interpreter for heatmaps.
            class_names (list, optional): List of class names for visualization.
            
        Returns:
            tuple: (test_loss, test_acc, precision, recall, f1, all_preds, all_labels)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        test_bar = tqdm(self.test_loader, desc="Testing") if self.config.USE_TQDM else self.test_loader
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_bar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                
                if isinstance(self.criterion, MixUpLoss):
                    loss = self.criterion(outputs, labels)  # Use default lam=1.0 for testing
                else:
                    loss = self.criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                test_bar.set_postfix(loss=test_loss/total, acc=100.*correct/total)

        test_loss = test_loss / len(self.test_loader.dataset)
        test_acc = 100. * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")

        if class_names:
            Utils.plot_confusion_matrix(all_labels, all_preds, class_names)

        return test_loss, test_acc, precision, recall, f1, all_preds, all_labels
