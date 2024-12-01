import os
import cv2
import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import kaggle
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from pathlib import Path
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class KITTIDataset(Dataset):
    """KITTI dataset with visualization and augmentation capabilities"""

    LABEL_COLORS = {
        'Car': (255,0,0),
        'Van': (255,255,0),
        'Truck': (255,255,255),
        'Pedestrian': (0,255,255),
        'Person_sitting': (0,255,255),
        'Cyclist': (0,128,255),
        'Tram': (128,0,0),
        'Misc': (0,255,255),
        'DontCare': (255,255,0)
    }

    @classmethod
    def download_and_prepare(cls, base_path="data"):
        """Download KITTI dataset using Kaggle API"""
        base_path = Path(base_path)
        dataset_path = base_path / "kitti-dataset"

        if not dataset_path.exists():
            print("Downloading KITTI dataset...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "klemenko/kitti-dataset",
                path=str(dataset_path),
                unzip=True
            )
            # setupkaggle()
            # subprocess.run(f'kaggle datasets download klemenko/kitti-dataset -d {dataset_path} --unzip')

        # Setup paths
        label_path = dataset_path / "data_object_label_2/training/label_2"
        image_path = dataset_path / "data_object_image_2/training/image_2"

        # Create DataFrame
        masks = sorted([f for f in os.listdir(label_path) if f.endswith('.txt')])
        df = pd.DataFrame({'masks': masks})
        df['images'] = df['masks'].str.replace('.txt', '.png')

        return str(image_path), str(label_path), df

    def __init__(self,
                 image_path: str,
                 label_path: str,
                 df: pd.DataFrame,
                 mode: str = 'train',
                 target_size: tuple = (256, 256),
                 augment: bool = True):
        self.image_path = Path(image_path)
        self.label_path = Path(label_path)
        self.df = df
        self.mode = mode
        self.target_size = target_size
        self.augment = augment and mode == 'train'
        self.num_classes = 34

        # Setup augmentation
        if self.augment:
            self.aug_seq = iaa.Sequential([
                iaa.Dropout([0.00, 0.06]),
                iaa.Sharpen((0.0, 0.1)),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                iaa.Affine(rotate=(-10, 10)),
                iaa.GammaContrast((0.5, 2.0))
            ])

    def __len__(self):
        return len(self.df)

    def create_mask(self, label_file, img_shape):
        """Create multi-class mask from label file"""
        mask = np.zeros(shape=(img_shape[0], img_shape[1]), dtype=np.int64)

        class_mapping = {
            'Car': 1,
            'Van': 2,
            'Truck': 3,
            'Pedestrian': 4,
            'Person_sitting': 5,
            'Cyclist': 6,
            'Tram': 7,
            'Misc': 8,
            'DontCare': 9
        }

        with open(label_file) as f:
            content = f.readlines()

        for line in content:
            data = line.split()
            label = data[0]
            if label in class_mapping:
                x1, y1, x2, y2 = map(int, map(float, data[4:8]))
                mask[y1:y2, x1:x2] = class_mapping[label]

        return mask

    def visualize_sample(self, idx, with_bbox=True, with_mask=True):
        """Visualize a sample with optional bbox and mask overlay"""
        img_file = self.image_path / self.df.iloc[idx]['images']
        label_file = self.label_path / self.df.iloc[idx]['masks']

        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        if with_bbox:
            img_bbox = image.copy()
            self.draw_boxes(img_bbox, label_file)
            axes[1].imshow(img_bbox)
            axes[1].set_title('Bounding Boxes')
            axes[1].axis('off')

        if with_mask:
            mask = self.create_mask(label_file, image.shape)
            img_mask = self.overlay_mask(image, mask)
            axes[2].imshow(img_mask)
            axes[2].set_title('Segmentation Mask')
            axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    def draw_boxes(self, image, label_file):
        """Draw bounding boxes on image"""
        with open(label_file) as f:
            content = f.readlines()

        for line in content:
            data = line.split()
            label = data[0]
            if label in self.LABEL_COLORS:
                x1, y1, x2, y2 = map(int, map(float, data[4:8]))
                color = self.LABEL_COLORS[label]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def overlay_mask(self, image, mask):
        """Overlay mask on image"""
        mask_rgb = np.repeat(mask, 3, axis=2)
        overlay = cv2.addWeighted(image, 0.8, (mask_rgb * 255).astype(np.uint8), 0.2, 0)
        return overlay

    def __getitem__(self, idx):
        img_file = self.image_path / self.df.iloc[idx]['images']
        label_file = self.label_path / self.df.iloc[idx]['masks']

        # Load image and create mask
        image = cv2.imread(str(img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.create_mask(label_file, image.shape)

        # Resize
        image = cv2.resize(image, self.target_size[::-1])
        mask = cv2.resize(mask, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.augment:
            segmap = SegmentationMapsOnImage(mask, shape=image.shape)
            image, segmap = self.aug_seq(image=image, segmentation_maps=segmap)
            mask = segmap.get_arr()

        # Convert to tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()  # Convert to long for CrossEntropyLoss

        # Normalize image
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(image)

        return image, mask



def create_kitti_dataloaders(base_path="data", batch_size=8, num_workers=4):
    """Create train, validation, and test dataloaders"""
    # Download and prepare dataset
    image_path, label_path, df = KITTIDataset.download_and_prepare(base_path)

    # Split dataset
    train_df = df.sample(frac=0.7, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    # Create datasets
    train_dataset = KITTIDataset(image_path, label_path, train_df, mode='train', augment=True)
    val_dataset = KITTIDataset(image_path, label_path, val_df, mode='val', augment=False)
    test_dataset = KITTIDataset(image_path, label_path, test_df, mode='test', augment=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
