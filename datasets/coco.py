import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

class CocoSegmentationDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform or self._default_transform()

        # Initialize COCO api for instance annotations
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Get all categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories.sort(key=lambda x: x['id'])

        # Background class + other classes
        self.num_classes = len(self.categories) + 1

        # Category id to continuous id mapping
        self.continuous_cat_id = {v['id']: i + 1 for i, v in enumerate(self.categories)}

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        # Load image
        img_id = self.ids[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        image = Image.open(os.path.join(self.root_dir, img_metadata['file_name']))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get all annotations for this image
        annot_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(annot_ids)

        # Create empty mask
        mask = np.zeros((img_metadata['height'], img_metadata['width']))

        # Fill mask with instance segmentations
        for annotation in annotations:
            # Get category label
            cat_id = annotation['category_id']
            class_id = self.continuous_cat_id[cat_id]

            # Get segmentation mask for this instance
            seg_mask = self.coco.annToMask(annotation)
            mask[seg_mask > 0] = class_id

        # Resize to fixed size
        image = image.resize((256, 256), Image.BILINEAR)
        mask = Image.fromarray(mask).resize((256, 256), Image.NEAREST)
        mask = np.array(mask)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()

        return image, mask

    def __len__(self):
        return len(self.ids)

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
    # Define paths
    train_data_dir = os.path.join(base_path, "train2017")
    val_data_dir = os.path.join(base_path, "val2017")
    train_annot_file = os.path.join(base_path, "annotations", "instances_train2017.json")
    val_annot_file = os.path.join(base_path, "annotations", "instances_val2017.json")

    # Create datasets
    train_dataset = CocoSegmentationDataset(
        root_dir=train_data_dir,
        annotation_file=train_annot_file
    )

    val_dataset = CocoSegmentationDataset(
        root_dir=val_data_dir,
        annotation_file=val_annot_file
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

    # Use validation set as test set
    test_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader