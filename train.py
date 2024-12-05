from pkgutil import get_loader
from sys import argv
from sklearn.metrics import pair_confusion_matrix
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
from torchmetrics import Dice, JaccardIndex, Precision, Recall, F1Score, ConfusionMatrix
from utils.visualization import SegmentationVisualizer
device ='cuda' if torch.cuda.is_available() else 'cpu'

def calculate_metrics(outputs, targets, n_classes, debug=False):
    """
    Calculate evaluation metrics for multiclass image segmentation.
    Args:
        outputs (torch.Tensor): Model predictions (B, C, H, W)
        targets (torch.Tensor): Ground truth labels (B, H, W)
        n_classes (int): Number of classes
    Returns:
        dict: Dictionary containing various metrics
    """
    # Get predictions
    preds = torch.argmax(outputs, dim=1)
    # valid_mask = targets != 255
    # preds = preds[valid_mask]
    # targets = targets[valid_mask]
    
    if debug:
        # Debug information
        print("\nDebugging Metrics Calculation:")
        print(f"Number of classes (n_classes): {n_classes}")
        print(f"Unique values in predictions: {torch.unique(preds).tolist()}")
        print(f"Unique values in targets: {torch.unique(targets).tolist()}")
        print(f"Max prediction value: {preds.max().item()}")
        print(f"Max target value: {targets.max().item()}")

    # Initialize metrics with task='multiclass'
    dice = Dice(num_classes=n_classes, average='macro').to(device)
    iou = JaccardIndex(task="multiclass", num_classes=n_classes, average='macro').to(device)
    precision = Precision(task="multiclass", num_classes=n_classes, average='macro').to(device)
    recall = Recall(task="multiclass", num_classes=n_classes, average='macro').to(device)
    f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro').to(device)
    confmat = ConfusionMatrix(task="multiclass", num_classes=n_classes).to(device)

    # Calculate metrics
    metrics = {
        'dice': dice(preds, targets).item(),
        'iou': iou(preds, targets).item(),
        'precision': precision(preds, targets).item(),
        'recall': recall(preds, targets).item(),
        'f1_score': f1(preds, targets).item(),
        'confusion_matrix': confmat(preds, targets).cpu().numpy()
    }
    torch.cuda.empty_cache()
    # Calculate per-class metrics
    targets = targets.long()
    per_class_metrics = {}
    dice_metric = Dice(average='micro').to(device)
    iou_metric = JaccardIndex(task="binary").to(device)

    for cls in range(n_classes):
        cls_preds = (outputs.argmax(dim=1) == cls).float().to(device)
        cls_targets = (targets == cls).float().to(device)

        if not cls_targets.any():
            continue
        cls_targets= cls_targets.long()
        cls_dice = dice_metric(cls_preds, cls_targets).item()
        cls_iou = iou_metric(cls_preds, cls_targets).item()

        per_class_metrics[f'class_{cls}_dice'] = cls_dice
        per_class_metrics[f'class_{cls}_iou'] = cls_iou

    metrics.update(per_class_metrics)
    return metrics

def validate(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss = 0
    all_outputs = []
    all_targets = []
    with tqdm(loader) as pbar:
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                # Store outputs and targets for later metric calculation
                all_outputs.append(outputs)
                all_targets.append(masks)
    print('All outputs and targets collected. Calculating metrics...')
    # Concatenate all batches
    outputs = torch.cat(all_outputs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Calculate metrics once for the entire epoch
    metrics = calculate_metrics(outputs, targets, n_classes)
    torch.cuda.empty_cache()
    return total_loss / len(loader), metrics

def validate_batch(model, loader, criterion, device, n_classes):
    model.eval()
    total_loss = 0

    # Initialize with empty metrics dictionary instead of None
    total_metrics = {
        'iou': 0,
        'dice': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0,
        'confusion_matrix': np.zeros((n_classes, n_classes))
    }
    # Initialize per-class metrics
    for cls in range(n_classes):
        total_metrics[f'class_{cls}_dice'] = 0
        total_metrics[f'class_{cls}_iou'] = 0

    batch_count = 0

    with tqdm(loader) as pbar:
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                # Calculate metrics for current batch
                masks = masks.long()
                batch_metrics = calculate_metrics(outputs, masks, n_classes)

                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if k != 'confusion_matrix':
                        total_metrics[k] = (total_metrics[k] * batch_count + v) / (batch_count + 1)
                total_metrics['confusion_matrix'] += batch_metrics['confusion_matrix']

                batch_count += 1

                # Update progress bar
                pbar.set_postfix({'val_loss': loss.item(), 
                                  'val_iou': total_metrics['iou'],
                                  'val_dice': total_metrics['dice'],
                                  'val_precision': total_metrics['precision'],
                                  'val_recall': total_metrics['recall'],
                                  'val_f1_score': total_metrics['f1_score']})
    torch.cuda.empty_cache()
    # Calculate average loss
    avg_loss = total_loss / len(loader)

    return avg_loss, total_metrics

def train_one_epoch(model, loader, criterion, optimizer, device, n_classes):
    model.train()
    total_loss = 0
    total_metrics = {
        'dice': 0,
        'iou': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }
    batch_count = 0
    for cls in range(n_classes):
        total_metrics[f'class_{cls}_dice'] = 0
        total_metrics[f'class_{cls}_iou'] = 0
    with tqdm(loader) as pbar:
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            masks = masks.long()
            batch_metrics = calculate_metrics(outputs, masks, n_classes)
            for k, v in batch_metrics.items():
                if k in total_metrics:
                    total_metrics[k] = (total_metrics[k] * batch_count + v) / (batch_count + 1)
                else:
                    total_metrics[k] = v
            total_loss += loss.item()

            # Update progress bar with just the loss
            pbar.set_postfix({'loss': loss.item(), 'train_dice': total_metrics['dice'],
                            'train_iou': total_metrics['iou'], 
                            'train_precision': total_metrics['precision'],
                            'train_recall': total_metrics['recall'],
                            'train_f1_score': total_metrics['f1_score']})
    torch.cuda.empty_cache()
    return total_loss / len(loader), total_metrics

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
    print('Device: ', device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    visualizer = SegmentationVisualizer(args.dataset)

    # Training loop
    best_val_iou = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train (now returns only loss)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, n_classes
        )

        # Validate (returns loss and metrics)
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, n_classes
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if args.use_wandb:
            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'val_{k}': v for k, v in val_metrics.items() if k != 'confusion_matrix'},
                'lr': optimizer.param_groups[0]['lr']
            })

            # Plot and log confusion matrix
            fig = visualizer.plot_confusion_matrix(val_metrics['confusion_matrix'], class_names=[f'class_{i}' for i in range(n_classes)])
            wandb.log({'val_confusion_matrix': wandb.Image(fig)})

        # Save best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['iou'],
            }, Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')

    # Test best model
    model.load_state_dict(torch.load(Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')['model_state_dict'])
    test_loss, test_metrics = validate(model, test_loader, criterion, device, n_classes)
    wandb.log({'test_loss': test_loss, 'test_iou': test_metrics['iou']})
    wandb.finish()
    print(f"\nTest Loss: {test_loss:.4f}, Test IoU: {test_metrics['iou']:.4f}")

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