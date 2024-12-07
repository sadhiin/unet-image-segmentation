import os
from tracemalloc import is_tracing
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv/monitor"
]

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
    [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
    [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
    [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
]

class VOCSegmentationDataset(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform or self._default_transform()

        # Read split file
        split_file = os.path.join(root, "ImageSets", "Segmentation", f"train.txt" if is_train else "val.txt")

        with open(split_file, mode='r') as f:
            image_names = [line.strip() for line in f.readlines()]

        self.images_path = [os.path.join(root, "JPEGImages", f"{name}.jpg") for name in image_names]
        self.masks_path = [os.path.join(root, "SegmentationClass", f"{name}.png") for name in image_names]

        # Verify files exist
        missing_files = []
        for img_path, mask_path in zip(self.images_path, self.masks_path):
            if not os.path.exists(img_path):
                missing_files.append(f"Missing image: {img_path}")
            if not os.path.exists(mask_path):
                missing_files.append(f"Missing mask: {mask_path}")

        if missing_files:
            print("\nMissing files:")
            for file in missing_files[:10]:  # Show first 10 missing files
                print(file)
            if len(missing_files) > 10:
                print(f"... and {len(missing_files) - 10} more")
            raise ValueError("Dataset files are missing")

        self.colormap2label = self._build_colormap2label()


    def _build_colormap2label(self):
        colormap2label = np.zeros(256 ** 3)
        for i, colormap in enumerate(VOC_COLORMAP):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
        return colormap2label

    def _convert_to_label(self, mask):
        data = mask.astype(np.int32)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return self.colormap2label[idx]

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        # img_name = self.images[idx]
        # mask_name = self.masks[idx]
        # print(f"Image name: {img_name}")
        # print(f"Mask name: {mask_name}")
        # Load image
        img_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        if mask is None:
            raise ValueError(f"Mask not found at {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Convert mask to label indices
        mask = self._convert_to_label(mask)

        # Resize to 256x256
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.images_path)

def create_voc_dataloaders(base_path, batch_size=8, num_workers=4):
    """Create train, validation and test dataloaders for Pascal VOC dataset."""
    data_dir = os.path.join(base_path, "pascal-voc")
    # Download dataset if not exists
    if not os.path.exists(data_dir):
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'gopalbhattrai/pascal-voc-2012-dataset',
            path = data_dir,
            unzip=True
        )
        # raise ValueError("Dataset not found")
    else:
        print(f"Dataset found at {data_dir}")

    # Create datasets
    train_dataset = VOCSegmentationDataset(
        root=os.path.join(data_dir,'VOC2012_train_val/VOC2012_train_val'),
        is_train=True
    )

    val_dataset = VOCSegmentationDataset(
        root=os.path.join(data_dir,'VOC2012_train_val/VOC2012_train_val'),
        is_train=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Use validation set as test set for Pascal VOC
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
