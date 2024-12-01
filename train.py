from sys import argv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from datasets.kitti import create_kitti_dataloaders
from datasets.citispace import create_cityscapes_dataloaders
from datasets.pascal_voc import create_voc_dataloaders
from datasets.coco import create_coco_dataloaders
from unet import UNet
from tqdm import tqdm
import wandb
import numpy as np

def calculate_iou(pred, target, n_classes):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection / union))

    return np.nanmean(ious)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    with tqdm(loader) as pbar:
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks, model.n_classes)

            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({'loss': loss.item(), 'iou': iou})

    return total_loss / len(loader), total_iou / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks, model.n_classes)

            total_loss += loss.item()
            total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)

def main(args):
    # Setup device
    device ='cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize wandb
    if args.use_wandb:
        wandb.login()
        wandb.init(project=f"Unet-semantic-segmentation", config=args)

    # Create datasets based on dataset choice
    if args.dataset.lower() == 'kitti':
        train_loader, val_loader, test_loader = create_kitti_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 10  # KITTI classes
        criterion = nn.CrossEntropyLoss()

    elif args.dataset.lower() == 'cityscapes':
        train_loader, val_loader, test_loader = create_cityscapes_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 21  # Cityscapes classes
        criterion = nn.CrossEntropyLoss(ignore_index=255)  # 255 is typically used for ignored labels

    elif args.dataset.lower() == 'pascal_voc':
        train_loader, val_loader, test_loader = create_voc_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 21  # Pascal VOC has 20 classes + background
        criterion = nn.CrossEntropyLoss(ignore_index=255)
    elif args.dataset.lower()=='coco':
        train_loader, val_loader, test_loader = create_coco_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 81  # COCO has 80 classes + background
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    else:
        raise ValueError(f"Undefined dataset: {args.dataset}")

    in_channels = 3


    # Initialize model with appropriate number of classes
    model = UNet(n_channels=in_channels, n_classes=n_classes).to(device)
    print("Model: \n")
    print(model)
    # Loss function and optimizer

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Training loop
    best_val_iou = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')



    # Test best model
    model.load_state_dict(torch.load(Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')['model_state_dict'])
    test_loss, test_iou = validate(model, test_loader, criterion, device)
    wandb.log({'test_loss': test_loss, 'test_iou': test_iou})
    wandb.finish()
    print(f"\nTest Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training')

    parser.add_argument('--dataset', type=str, default='kitti',
                    help='Dataset to use (kitti, cityscapes, pascal_voc, or coco)')
    parser.add_argument('--data_dir', type=str, default='data',
                    help='Base directory for datasets')


    parser.add_argument('--image_size', type=int, default=256,
                      help='Size to resize images to')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')

    # Other parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                      help='Whether to use Weights & Biases logging')

    args = parser.parse_args()

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    main(args)