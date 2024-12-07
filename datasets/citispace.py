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

    def __len__(self):
        return len(self.image_fns)


    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp).convert('RGB')
        image = np.array(image)

        # Split the combined image into cityscape and label
        cityscape, label = image[:,:int(image.shape[1]/2)],image[:,int(image.shape[1]/2):]

        if self.transform:
            cityscape = self.transform(Image.fromarray(cityscape))

        # Convert label to tensor
        label = torch.from_numpy(label).long()

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
        print('Downloading the Citispace datasets')
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
                "dansbecker/cityscapes-image-pairs",
                path=str(base_path),
                unzip=True
            )
        data_dir = os.path.join(base_path, 'cityscapes_data')
        raise ValueError("Citispace data is not downloaded. Please download the data.")
    else:
        print("Citispace data is previously downloaded...!")
        data_dir = os.path.join(base_path, 'cityscapes_data')

    train_images = os.listdir(os.path.join(data_dir, "train"))
    val_images = os.listdir(os.path.join(data_dir,'val'))
    test_images = val_images[:int(len(val_images)/2)]
    val_images = val_images[int(len(val_images)/2):]
    np.random.seed(42)
    np.random.shuffle(train_images)
    np.random.shuffle(val_images)

    train_dataset = CityscapesDataset(
        os.path.join(data_dir, "train"),
        image_list=train_images
    )
    val_dataset = CityscapesDataset(
        os.path.join(data_dir, "val"),
        image_list=val_images
    )
    test_dataset = CityscapesDataset(
        os.path.join(data_dir, "val"),
        image_list=test_images
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

    return train_loader, val_loader, test_loader


