import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

class CocoSegmentationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, is_train=True,
                 max_cache_size=1000, image_size=256, pregenerate_masks=False):
        """
        Args:
            root_dir: Directory with all the images
            annotation_file: Path to COCO annotation file
            transform: Optional transform to be applied
            is_train: Whether this is training set (for augmentations)
            max_cache_size: Maximum number of masks to cache
            image_size: Size to resize images to
            pregenerate_masks: Whether to pregenerate all masks
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        self.transform = transform or self._default_transform()
        self.max_cache_size = max_cache_size

        # Initialize COCO api
        try:
            self.coco = COCO(annotation_file)
            self.ids = list(sorted(self.coco.imgs.keys()))
        except Exception as e:
            raise RuntimeError(f"Error loading COCO annotations: {str(e)}")

        # Setup categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories.sort(key=lambda x: x['id'])
        self.num_classes = len(self.categories) + 1  # +1 for background
        self.continuous_cat_id = {v['id']: i + 1 for i, v in enumerate(self.categories)}

        # Setup mask caching
        self.mask_dir = Path(os.path.dirname(root_dir)) / 'masks'
        self.mask_dir.mkdir(parents=True, exist_ok=True)

        # Pregenerate masks if requested
        if pregenerate_masks:
            self.generate_all_masks()

        # Clean old cache if necessary
        self._clean_old_cache()

    def _default_transform(self):
        """Default transformation with optional augmentation"""
        if self.is_train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
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

    def _clean_old_cache(self):
        """Clean old cache files if exceeding max size"""
        cache_files = list(self.mask_dir.glob('*.npy'))
        if len(cache_files) > self.max_cache_size:
            cache_files.sort(key=lambda x: x.stat().st_mtime)
            for f in cache_files[:-self.max_cache_size]:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Error removing cache file {f}: {str(e)}")

    def generate_all_masks(self):
        """Pre-generate all masks"""
        print("Pre-generating masks...")
        for img_id in tqdm(self.ids, desc="Generating masks"):
            img_metadata = self.coco.loadImgs(img_id)[0]
            self._load_or_generate_mask(
                img_id,
                img_metadata['height'],
                img_metadata['width']
            )

    def _generate_and_save_mask(self, img_id, height, width):
        """Generate mask and save it to disk"""
        try:
            # Create empty mask
            mask = np.zeros((height, width), dtype=np.uint8)  # Specify dtype

            # Get all annotations for this image
            annot_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(annot_ids)

            # Fill mask with instance segmentations
            for annotation in annotations:
                cat_id = annotation['category_id']
                class_id = self.continuous_cat_id[cat_id]
                seg_mask = self.coco.annToMask(annotation)
                mask[seg_mask > 0] = class_id

            # Save mask to file
            mask_path = self.mask_dir / f'{img_id}.npy'
            np.save(str(mask_path), mask)  # Convert Path to string

            return mask
        except Exception as e:
            print(f"Error generating mask for image {img_id}: {str(e)}")
            return np.zeros((height, width), dtype=np.uint8)

    def _load_or_generate_mask(self, img_id, height, width):
        """Load mask from cache if it exists, otherwise generate and cache it"""
        mask_path = self.mask_dir / f'{img_id}.npy'

        try:
            if mask_path.exists():
                try:
                    mask = np.load(str(mask_path))  # Convert Path to string
                    if mask.shape == (height, width):
                        return mask
                    else:
                        print(f"Cached mask shape mismatch for image {img_id}. Regenerating...")
                except Exception as e:
                    print(f"Error loading cached mask for image {img_id}: {str(e)}")

            # Generate new mask if loading failed or shapes don't match
            return self._generate_and_save_mask(img_id, height, width)

        except Exception as e:
            print(f"Error in mask loading/generation for image {img_id}: {str(e)}")
            return np.zeros((height, width))

    def __getitem__(self, index):
        try:
            # Load image
            img_id = self.ids[index]
            img_metadata = self.coco.loadImgs(img_id)[0]
            image_path = os.path.join(self.root_dir, img_metadata['file_name'])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = Image.open(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Load or generate mask with explicit dimensions
            original_height = img_metadata['height']
            original_width = img_metadata['width']

            mask = self._load_or_generate_mask(
                img_id,
                original_height,
                original_width
            )

            # Verify mask was loaded/generated correctly
            if mask is None or mask.shape != (original_height, original_width):
                print(f"Invalid mask shape for image {img_id}. Expected {(original_height, original_width)}, got {mask.shape if mask is not None else None}")
                mask = np.zeros((original_height, original_width), dtype=np.uint8)

            # Resize to fixed size
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            mask = Image.fromarray(mask.astype(np.uint8)).resize((self.image_size, self.image_size), Image.NEAREST)
            mask = np.array(mask)

            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)

            # Convert mask to tensor
            mask = torch.from_numpy(mask).long()

            return image, mask

        except Exception as e:
            print(f"Error loading sample {index}: {str(e)}")
            # Return a dummy sample
            return torch.zeros((3, self.image_size, self.image_size)), torch.zeros((self.image_size, self.image_size), dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def calculate_dataset_stats(self):
        """Calculate dataset statistics for normalization"""
        means = []
        stds = []
        print("Calculating dataset statistics...")
        for idx in tqdm(range(min(1000, len(self))), desc="Computing stats"):
            image, _ = self.__getitem__(idx)
            means.append(image.mean(dim=[1, 2]))
            stds.append(image.std(dim=[1, 2]))

        mean = torch.stack(means).mean(dim=0)
        std = torch.stack(stds).mean(dim=0)
        return mean.tolist(), std.tolist()

def create_coco_dataloaders(base_path, batch_size=8, num_workers=4):
    """Create train, validation and test dataloaders for COCO dataset."""
    if not os.path.exists(os.path.join(base_path,'coco2017')):
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'awsaf49/coco-2017-dataset',
            path = base_path,
            unzip=True
        )
        base_path = os.path.join(base_path,'coco2017')
    else:
        base_path = os.path.join(base_path,'coco2017')

    # Define paths
    train_data_dir = os.path.join(base_path, "train2017")
    val_data_dir = os.path.join(base_path, "val2017")
    train_annot_file = os.path.join(base_path, "annotations", "instances_train2017.json")
    val_annot_file = os.path.join(base_path, "annotations", "instances_val2017.json")

    # Create datasets with different transforms for train/val
    train_dataset = CocoSegmentationDataset(
        root_dir=train_data_dir,
        annotation_file=train_annot_file,
        is_train=True,
        pregenerate_masks=True  # Pre-generate masks for training set
    )

    val_dataset = CocoSegmentationDataset(
        root_dir=val_data_dir,
        annotation_file=val_annot_file,
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

    # Use validation set as test set
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader