import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as TF

class SegmentationVisualizer:
    def __init__(self, dataset_name, base_dir='visualizations'):
        """
        Initialize visualizer for segmentation datasets.
        Args:
            dataset_name: Name of the dataset (e.g., 'coco', 'pascal_voc', 'cityscapes', 'kitti')
            base_dir: Base directory for saving visualizations
        """
        self.dataset_name = dataset_name.lower()
        self.base_dir = Path(base_dir) / dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Validate dataset name
        if self.dataset_name not in ['coco', 'pascal_voc', 'cityscapes', 'kitti', 'spacenet']:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Define color maps for different datasets
        self.color_maps = {
            'coco': 'tab20',
            'pascal_voc': 'tab20',
            'cityscapes': 'tab20',
            'kitti': 'tab20',
            'spacenet': 'binary'
        }

        # Define normalization parameters for different datasets
        self.norm_params = {
            'coco': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'pascal_voc': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'cityscapes': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'kitti': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'spacenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        }

    def denormalize_image(self, image):
        """Denormalize image tensor to [0,1] range"""
        if not isinstance(image, torch.Tensor):
            raise TypeError("Image must be a torch.Tensor")

        norm_params = self.norm_params[self.dataset_name]
        # Ensure the image tensor is in the correct shape (C, H, W)
        if image.ndim == 3 and image.shape[0] == 3:
            return TF.normalize(
                image,
                mean=[-m/s for m, s in zip(norm_params['mean'], norm_params['std'])],
                std=[1/s for s in norm_params['std']]
            )
        else:
            raise ValueError(f"Expected image tensor of shape (3, H, W), but got {image.shape}")

    def visualize_batch(self, images, masks, predictions=None, phase='train', batch_idx=0, max_samples=4):
        """
        Visualize a batch of images, masks, and optionally predictions.
        Args:
            images: Tensor of shape (B, C, H, W)
            masks: Tensor of shape (B, H, W)
            predictions: Optional tensor of shape (B, C, H, W)
            phase: One of 'train', 'val', or 'test'
            batch_idx: Batch index for saving files
            max_samples: Maximum number of samples to visualize from the batch
        """
        if phase not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid phase: {phase}")

        save_dir = self.base_dir / phase
        save_dir.mkdir(exist_ok=True)

        try:
            # Move tensors to CPU and convert to numpy
            images = images.detach().cpu()
            masks = masks.detach().cpu()
            if predictions is not None:
                predictions = predictions.detach().cpu()
                predictions = torch.argmax(predictions, dim=1)

            # Process each image in the batch
            for i in range(min(images.shape[0], max_samples)):
                # Create figure
                fig, axes = plt.subplots(1, 3 if predictions is not None else 2,
                                    figsize=(15, 5))

                # Plot original image
                img = images[i].permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
                img = self.denormalize_image(img.permute(2, 0, 1))  # Denormalize
                img = img.permute(1, 2, 0).numpy()  # Convert to numpy for plotting
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Plot ground truth mask
                mask = masks[i].numpy()
                if self.dataset_name == 'spacenet':
                    axes[1].imshow(mask, cmap='binary')
                else:
                    axes[1].imshow(mask, cmap=self.color_maps[self.dataset_name])
                axes[1].set_title('Ground Truth Mask')
                axes[1].axis('off')

                # Plot prediction if available
                if predictions is not None:
                    pred = predictions[i].numpy()
                    if self.dataset_name == 'spacenet':
                        axes[2].imshow(pred, cmap='binary')
                    else:
                        axes[2].imshow(pred, cmap=self.color_maps[self.dataset_name])
                    axes[2].set_title('Predicted Mask')
                    axes[2].axis('off')

                # Save the figure
                plt.tight_layout()
                save_path = save_dir / f'batch_{batch_idx}_sample_{i}.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            plt.close('all')  # Clean up any open figures

    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot the confusion matrix.
        Args:
            cm: Confusion matrix
            class_names: List of class names
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        cax = ax.matshow(cm, cmap='Blues')
        plt.colorbar(cax)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Loop over data dimensions and create text annotations.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")

        plt.tight_layout()
        return fig