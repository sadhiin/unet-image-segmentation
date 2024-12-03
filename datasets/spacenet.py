# https://github.com/Rim-chan/SpaceNet7-Buildings-Detection/blob/main/README.rst
# https://github.com/reachsumit/deep-unet-for-satellite-image-segmentation
import os
import torch
import numpy as np
import skimage.io as io
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import rasterio

class SpaceNetDataset(Dataset):
    def __init__(self, files, crop_size=256, transform=None, is_train=True):
        """
        Args:
            files: List of dictionaries containing image and mask paths
            crop_size: Size of the random crop during training
            transform: Optional transform to be applied
            is_train: Whether this is training set (for augmentations)
        """
        self.files = files
        self.crop_size = crop_size
        self.is_train = is_train
        self.transform = transform or self._default_transform()

    def __len__(self):
        return len(self.files)

    def _default_transform(self):
        """Default transformation with optional augmentation"""
        if self.is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        try:
            # Load image and mask
            image = self._load_image(idx)
            mask = self._load_mask(idx)

            # Convert to PIL for transforms
            image = Image.fromarray(image)

            # Apply random crop if training
            if self.is_train:
                image, mask = self._random_crop(image, mask)
            else:
                # Resize to fixed size for validation/testing
                image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                mask = Image.fromarray(mask).resize((self.crop_size, self.crop_size), Image.NEAREST)
                mask = np.array(mask)

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            # Convert mask to tensor
            mask = torch.from_numpy(mask).long()

            return image, mask

        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return torch.zeros((3, self.crop_size, self.crop_size)), \
                   torch.zeros((self.crop_size, self.crop_size), dtype=torch.long)

    def _load_image(self, idx):
        """Load and normalize image"""
        image = io.imread(self.files[idx]['image'])[:,:,:3]
        return (image / np.iinfo(image.dtype).max).astype(np.float32)

    def _load_mask(self, idx):
        """Load and binarize mask using rasterio"""
        try:
            with rasterio.open(self.files[idx]['mask']) as src:
                mask = src.read(1)  # Read first band
                # Ensure mask is properly oriented
                if src.transform.e > 0:  # Check if we need to flip vertically
                    mask = np.flipud(mask)
                return np.where(mask > 0, 1, 0).astype(np.uint8)
        except Exception as e:
            print(f"Error loading mask {self.files[idx]['mask']}: {str(e)}")
            return np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)

    def _random_crop(self, image, mask, size=None):
        """Random crop both image and mask"""
        if size is None:
            size = self.crop_size

        # Get dimensions
        w, h = image.size
        th, tw = size, size

        if w == tw and h == th:
            return image, mask

        # Random crop
        x1 = np.random.randint(0, w - tw)
        y1 = np.random.randint(0, h - th)

        image = image.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask[y1:y1 + th, x1:x1 + tw]

        return image, mask

def create_spacenet_dataloaders(base_path, batch_size=8, num_workers=4):
    """Create train, validation and test dataloaders for SpaceNet dataset."""

    # Download dataset if not exists
    if not os.path.exists(os.path.join(base_path, 'spacenet')):
        print("Downloading SpaceNet dataset...")
        import boto3
        s3 = boto3.client('s3')

        try:
            # Download from S3
            bucket = 'spacenet-dataset'
            prefix = 'spacenet/SN7_buildings/'

            # Create directory
            os.makedirs(os.path.join(base_path, 'spacenet'), exist_ok=True)

            # Download files
            for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)['Contents']:
                key = obj['Key']
                local_path = os.path.join(base_path, 'spacenet', os.path.basename(key))
                s3.download_file(bucket, key, local_path)
                print(f"Downloaded {key}")

        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            return None, None, None

    # Get all files
    data_dir = os.path.join(base_path, 'spacenet')
    files = []

    for img_path in Path(data_dir).glob('**/images/*.tif'):
        mask_path = str(img_path).replace('images', 'masks').replace('.tif', '_mask.tif')
        if os.path.exists(mask_path):
            files.append({
                'image': str(img_path),
                'mask': mask_path
            })

    # Split into train, val, test
    train_files, temp_files = train_test_split(files, train_size=0.8, random_state=42)
    val_files, test_files = train_test_split(temp_files, train_size=0.5, random_state=42)

    # Create datasets
    train_dataset = SpaceNetDataset(
        files=train_files,
        is_train=True
    )

    val_dataset = SpaceNetDataset(
        files=val_files,
        is_train=False
    )

    test_dataset = SpaceNetDataset(
        files=test_files,
        is_train=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader