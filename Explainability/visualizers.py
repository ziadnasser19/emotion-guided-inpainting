import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients, GradCAM, LayerGradCAM, DeepLift, Saliency
from captum.attr import visualization as viz
import os
class ExplainabilityAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.ig = IntegratedGradients(self.model)
        self.dl = DeepLift(self.model)
        self.saliency = Saliency(self.model)
        self.gradcam = None
        self.class_names = ['Happy', 'Sad', 'Surprise', 'Neutral']

    def initialize_gradcam(self, model_name):
        if model_name in ["resnet18", "resnet50"]:
            target_layer = self.model.backbone.embedding_model.features[-1][-1]
            self.gradcam = LayerGradCAM(self.model, target_layer)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def to_rgb(self, attr):
        """
        Converts 2D or single-channel attribution maps to 3-channel RGB for visualization.
        """
        if attr.ndim == 2:
            return np.stack([attr]*3, axis=-1)
        elif attr.ndim == 3 and attr.shape[0] == 1:  # [1, H, W]
            attr = np.squeeze(attr, axis=0)
            return np.stack([attr]*3, axis=-1)
        return attr  # already 3-channel

    def generate_attributions(self, input_image, target_class, method='integrated_gradients', model_name='resnet18'):
        input_image = input_image.to(self.device)
        input_image.requires_grad = True
        if method == 'integrated_gradients':
            attributions, delta = self.ig.attribute(input_image, baselines=torch.zeros_like(input_image),
                                                  target=target_class, return_convergence_delta=True)
            return attributions, delta
        elif method == 'gradcam':
            if self.gradcam is None:
                self.initialize_gradcam(model_name)
            attributions = self.gradcam.attribute(input_image, target=target_class)
            return attributions, None
        elif method == 'deeplift':
            baselines = torch.zeros_like(input_image)
            attributions = self.dl.attribute(input_image, baselines=baselines, target=target_class,
                                            return_convergence_delta=False)
            return attributions, None
        elif method == 'saliency':
            attributions = self.saliency.attribute(input_image, target=target_class)
            return attributions, None
        else:
            raise ValueError(f"Unsupported method: {method}")

    def visualize_attributions(self, input_image, attributions, class_name, method, save_path=None, show=True):
        input_image = input_image.cpu().squeeze().numpy()
        if input_image.ndim == 2:
            input_image = np.stack([input_image]*3, axis=-1)
        elif input_image.shape[0] == 1:
            input_image = np.transpose(input_image, (1, 2, 0))
            input_image = np.repeat(input_image, 3, axis=2)
        input_image = (input_image * 0.5 + 0.5)  # Unnormalize
        attributions = self.to_rgb(attributions.cpu().squeeze().numpy())
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        viz.visualize_image_attr(None, input_image, method="original_image", title="Original",
                                plt_fig_axis=(fig, ax[0]), use_pyplot=False)
        viz.visualize_image_attr(attributions, input_image, method="blended_heat_map", sign="all",
                                show_colorbar=True, title=f"{method.capitalize()} ({class_name})",
                                plt_fig_axis=(fig, ax[1]), use_pyplot=False)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close(fig)

    def analyze_samples(self, dataloader, num_samples=5, sample_indices=None, method='integrated_gradients',
                       model_name='resnet18', save_dir=None):
        self.model.eval()
        images, labels = next(iter(dataloader))
        if sample_indices:
            indices = sample_indices[:num_samples]
        else:
            indices = np.random.choice(len(images), num_samples, replace=False)
        for idx in indices:
            img, label = images[idx:idx+1], labels[idx].item()
            img = img.to(self.device)
            output = self.model(img)
            pred = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, 1)[0, pred].item()
            true_class = self.class_names[label]
            pred_class = self.class_names[pred]
            print(f"Image {idx} - True: {true_class} | Predicted: {pred_class} | Prob: {prob:.4f}")
            attributions, delta = self.generate_attributions(img, pred, method, model_name)
            save_path = os.path.join(save_dir, f"attribution_{method}_{idx}.png") if save_dir else None
            self.visualize_attributions(img, attributions, pred_class, method, save_path, show=True)

    def analyze_samples_all_methods(self, dataloader, num_samples=5, sample_indices=None, model_name='resnet18', save_dir=None):
        self.model.eval()
        images, labels = next(iter(dataloader))
        if sample_indices:
            indices = sample_indices[:num_samples]
        else:
            indices = np.random.choice(len(images), num_samples, replace=False)
        for idx in indices:
            img, label = images[idx:idx+1], labels[idx].item()
            img = img.to(self.device)
            output = self.model(img)
            pred = torch.argmax(output, dim=1).item()
            prob = F.softmax(output, 1)[0, pred].item()
            true_class = self.class_names[label]
            pred_class = self.class_names[pred]
            print(f"Image {idx} - True: {true_class} | Predicted: {pred_class} | Prob: {prob:.4f}")
            original_image = img.cpu().squeeze().numpy()
            if original_image.ndim == 2:
                original_image = np.stack([original_image]*3, axis=-1)
            elif original_image.shape[0] == 1:
                original_image = np.transpose(original_image, (1, 2, 0))
                original_image = np.repeat(original_image, 3, axis=2)
            original_image = (original_image * 0.5 + 0.5)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            viz.visualize_image_attr(None, original_image, method="original_image", title="Original",
                                    plt_fig_axis=(fig, axes[0]), use_pyplot=False)
            for i, method in enumerate(['saliency', 'integrated_gradients', 'deeplift']):
                attributions, _ = self.generate_attributions(img, pred, method, model_name)
                if attributions is None:
                    print(f"Attribution failed for method: {method}")
                    continue
                attributions = self.to_rgb(attributions.cpu().squeeze().numpy())
                viz.visualize_image_attr(attributions, original_image, method="blended_heat_map", sign="all",
                                        show_colorbar=True, title=method.capitalize(),
                                        plt_fig_axis=(fig, axes[i+1]), use_pyplot=False)
            fig.suptitle(f"Image {idx} | True: {true_class} | Pred: {pred_class} ({prob:.2f})", fontsize=16)
            save_path = os.path.join(save_dir, f"attribution_all_{idx}.png") if save_dir else None
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
            plt.close(fig)