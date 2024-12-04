import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, image_dir, image_list=None, transform=None):
        self.image_dir = image_dir
        self.image_fns = image_list if image_list is not None else os.listdir(image_dir)
        self.transform = transform or self._default_transform()

        # Define the label mapping
        self.id_to_trainid = {
            0: 0,       # background
            1: 1,       # road
            2: 2,       # sidewalk
            3: 3,       # building
            4: 4,       # wall
            5: 5,       # fence
            6: 6,       # pole
            7: 7,       # traffic light
            8: 8,       # traffic sign
            9: 9,       # vegetation
            10: 10,     # terrain
            11: 11,     # sky
            12: 12,     # person
            13: 13,     # rider
            14: 14,     # car
            15: 15,     # truck
            16: 16,     # bus
            17: 17,     # train
            18: 18,     # motorcycle
            19: 19,     # bicycle
            255: 255    # ignore
        }
    def __len__(self):
        return len(self.image_fns)


    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)

        # Split the combined image into cityscape and label
        cityscape, label = self._split_image(image)

        # Map labels to training IDs
        label = self._map_labels(label)

        if self.transform:
            cityscape = self.transform(Image.fromarray(cityscape))

        # Convert label to tensor
        label = torch.from_numpy(label).long()

        # print(f"\nDebugging CityscapesDataset item {index}:")
        # print(f"Unique values in label: {torch.unique(label).tolist()}")
        # print(f"Label shape: {label.shape}")

        return cityscape, label

    def _map_labels(self, label):
        """Map labels to training IDs and ensure ignore class exists"""
        label_copy = np.zeros(label.shape[:2], dtype=np.int64)

        # Map known classes
        for k, v in self.id_to_trainid.items():
            label_copy[label[:,:,0] == k] = v

        # Ensure there's at least one ignore pixel (in a non-visible corner)
        if 255 not in np.unique(label_copy):
            label_copy[0, 0] = 255  # Set top-left pixel to ignore class

        return label_copy

    def _split_image(self, image):
        cityscape = image[:, :256, :]
        label = image[:, 256:, :]
        return cityscape, label

    def _default_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

def create_cityscapes_dataloaders(base_path, batch_size=8, num_workers=4):
    # Load all image files

    if not os.path.exists(os.path.join(base_path, 'cityscapes_data')):
        # print('Downloading the Citispace datasets')
        # import kaggle
        # kaggle.api.authenticate()
        # kaggle.api.dataset_download_files(
        #         "dansbecker/cityscapes-image-pairs",
        #         path=str(base_path),
        #         unzip=True
        #     )
        # data_dir = os.path.join(base_path, 'cityscapes_data')
        raise ValueError("Citispace data is not downloaded. Please download the data.")
    else:
        print("Citispace data is previously downloaded...!")
        data_dir = os.path.join(base_path, 'cityscapes_data')

    train_images = os.listdir(os.path.join(data_dir, "train"))
    val_images = os.listdir(os.path.join(data_dir,'val'))
    train_images = os.listdir(os.path.join(data_dir, "train"))

    # Split into train, val, test (70%, 15%, 15%)
    train_imgs, _ = train_test_split(train_images, train_size=0.01,shuffle=True ,random_state=42)
    val_imgs, test_imgs = train_test_split(val_images, train_size=0.5, shuffle=True,random_state=42)

    # Create datasets with specific image lists
    train_dataset = CityscapesDataset(
        os.path.join(data_dir, "train"),
        image_list=train_imgs
    )
    val_dataset = CityscapesDataset(
        os.path.join(data_dir, "val"),
        image_list=val_imgs
    )
    test_dataset = CityscapesDataset(
        os.path.join(data_dir, "val"),
        image_list=test_imgs
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # Debug: Print unique label values
    # sample_batch = next(iter(train_loader))
    # images, labels = sample_batch
    # print(f"Unique label values: {torch.unique(labels)}")
    # print(f"Label shape: {labels.shape}")
    # print(f"Min label value: {labels.min()}")
    # print(f"Max label value: {labels.max()}")

    return train_loader, val_loader, test_loader


