from asyncio import tasks
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
from datasets.spacenet import create_spacenet_dataloaders
from unet import UNet
from tqdm import tqdm
import wandb
import numpy as np
from torchmetrics import Dice, JaccardIndex, Precision, Recall, F1Score, ConfusionMatrix
from utils.visualization import SegmentationVisualizer
device ='cuda' if torch.cuda.is_available() else 'cpu'

def calculate_metrics(outputs, targets, n_classes):
    """Calculate segmentation metrics including per-class metrics"""
    # Move tensors to CPU for metric calculation
    outputs = outputs.detach().cpu()
    targets = targets.detach().cpu()

    # Get predictions
    predictions = torch.argmax(outputs, dim=1)

    # Initialize metrics dictionary
    metrics = {}

    # Calculate confusion matrix
    conf_matrix = torch.zeros((n_classes, n_classes))
    for t, p in zip(targets.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    metrics['confusion_matrix'] = conf_matrix.numpy()

    # Calculate global metrics first
    total_intersection = 0
    total_union = 0

    # Calculate per-class metrics only for classes present in this batch
    present_classes = torch.unique(targets)
    for class_id in range(n_classes):
        # Create binary masks for current class
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        # Only calculate metrics if class is present in targets
        if class_id in present_classes:
            intersection = torch.logical_and(pred_mask, target_mask).sum().item()
            union = torch.logical_or(pred_mask, target_mask).sum().item()

            # Avoid division by zero
            if union > 0:
                iou = intersection / union
                dice = 2 * intersection / (pred_mask.sum().item() + target_mask.sum().item() + 1e-10)
            else:
                iou = 0.0
                dice = 0.0

            metrics[f'class_{class_id}_iou'] = iou
            metrics[f'class_{class_id}_dice'] = dice

            total_intersection += intersection
            total_union += union

    # Calculate mean metrics only for classes that appeared
    metrics['iou'] = total_intersection / (total_union + 1e-10)
    metrics['dice'] = 2 * total_intersection / (predictions.numel() + targets.numel() + 1e-10)

    return metrics

def average_metrics(all_metrics, n_batches):
    """
    Average metrics across batches, handling missing classes
    """
    avg_metrics = {}
    class_metrics_count = {}

    # First pass: collect all metric keys and initialize counters
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())

    # Second pass: sum up metrics and count occurrences
    for key in all_keys:
        if key == 'confusion_matrix':
            # Sum up confusion matrices
            avg_metrics[key] = sum(m[key] for m in all_metrics if key in m)
        else:
            # For other metrics, average only when present
            values = [m[key] for m in all_metrics if key in m]
            if values:  # Only average if we have values
                avg_metrics[key] = sum(values) / len(values)
            else:
                avg_metrics[key] = 0.0  # Default value for missing metrics

    return avg_metrics

def validate(model, loader, criterion, device, n_classes, visualizer, epoch):
    model.eval()
    total_loss = 0
    all_metrics = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks, n_classes)
            all_metrics.append(batch_metrics)

            total_loss += loss.item()

            # Visualize first batch
            if batch_idx == 0:
                visualizer.visualize_batch(images, masks, outputs,
                                        phase='val', batch_idx=epoch)

    # Average metrics across batches
    avg_metrics = {}
    n_batches = len(all_metrics)

    for metric in all_metrics[0].keys():
        if metric != 'confusion_matrix':
            avg_metrics[metric] = sum(m[metric] for m in all_metrics) / n_batches

    # Sum confusion matrices
    avg_metrics['confusion_matrix'] = sum(m['confusion_matrix'] for m in all_metrics)

    return total_loss / len(loader), avg_metrics

def train_one_epoch(model, train_loader, criterion, optimizer, device, n_classes, visualizer, epoch):
    model.train()
    running_loss = 0.0
    all_metrics = []

    with tqdm(train_loader) as pbar:
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            # Calculate metrics [issue]
            batch_metrics = calculate_metrics(outputs, masks, n_classes)
            all_metrics.append(batch_metrics)

            running_loss += loss.item()

            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': batch_metrics['dice'],
                'iou': batch_metrics['iou']
            })

            # Visualize first batch
            if batch_idx == 0:
                visualizer.visualize_batch(images, masks, outputs,
                                        phase='train', batch_idx=epoch)

    # Average metrics across batches
    avg_metrics = average_metrics(all_metrics, len(train_loader))

    return running_loss / len(train_loader), avg_metrics

def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix'):
    """Plot confusion matrix using matplotlib."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.close(fig)
    return fig

def main(args):
    # Create datasets based on dataset choice first to get n_classes
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
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    elif args.dataset.lower() == 'pascal_voc':
        train_loader, val_loader, test_loader = create_voc_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 21  # Pascal VOC has 20 classes + background
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    elif args.dataset.lower() == 'coco':
        train_loader, val_loader, test_loader = create_coco_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 81  # COCO has 80 classes + background
        criterion = nn.CrossEntropyLoss(ignore_index=255)

    elif args.dataset.lower() == 'spacenet':
        train_loader, val_loader, test_loader = create_spacenet_dataloaders(
            base_path=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        n_classes = 2  # Binary segmentation for buildings
        criterion = nn.BCEWithLogitsLoss()  # Better for binary segmentation

    else:
        raise ValueError(f"Undefined dataset: {args.dataset}")

    # Setup device and WandB after n_classes is defined
    if args.use_wandb:
        wandb.login()
        # Enhanced WandB config
        wandb_config = {
            **vars(args),
            'device': device,
            'model_type': 'UNet',
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'dataset_classes': n_classes,
        }
        wandb.init(
            project=f"Unet-semantic-segmentation",
            name=f"{args.dataset}-{wandb.util.generate_id()}",
            config=wandb_config
        )

    # Initialize model with appropriate number of classes
    model = UNet(n_channels=3, n_classes=n_classes).to(device)
    print("Model: \n")
    print(model)

    if args.use_wandb:
        # Log model architecture
        wandb.watch(model, log='all', log_freq=100)

    # Loss function and optimizer

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Initialize visualizer
    visualizer = SegmentationVisualizer(args.dataset)

    # Training loop
    best_val_iou = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, n_classes,
            visualizer, epoch
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, n_classes,
            visualizer, epoch
        )

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        if args.use_wandb:
            # Enhanced metrics logging
            wandb.log({
                'epoch': epoch,
                'learning_rate': current_lr,

                # Loss metrics
                'train/loss': train_loss,
                'val/loss': val_loss,

                # Training metrics
                **{f'train/metrics/{k}': v for k, v in train_metrics.items()
                   if k != 'confusion_matrix'},

                # Validation metrics
                **{f'val/metrics/{k}': v for k, v in val_metrics.items()
                   if k != 'confusion_matrix'},

                # Per-class metrics (if available)
                **{f'train/class_metrics/{k}': v for k, v in train_metrics.items()
                   if k.startswith('class_')},
                **{f'val/class_metrics/{k}': v for k, v in val_metrics.items()
                   if k.startswith('class_')},

                # Learning rate
                'train/learning_rate': current_lr,
            })

            # Log confusion matrices and visualizations
            wandb.log({
                'train/confusion_matrix': wandb.Image(
                    plot_confusion_matrix(
                        train_metrics['confusion_matrix'],
                        title=f'Train Confusion Matrix - Epoch {epoch}'
                    )
                ),
                'val/confusion_matrix': wandb.Image(
                    plot_confusion_matrix(
                        val_metrics['confusion_matrix'],
                        title=f'Val Confusion Matrix - Epoch {epoch}'
                    )
                ),
                'train/samples': [wandb.Image(str(p)) for p in (visualizer.base_dir / 'train').glob('*.png')],
                'val/samples': [wandb.Image(str(p)) for p in (visualizer.base_dir / 'val').glob('*.png')]
            })

        # Save best model with more information
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_loss': val_loss,
                'train_loss': train_loss,
                'args': vars(args)
            }
            save_path = Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth'
            torch.save(checkpoint, save_path)

            if args.use_wandb:
                wandb.log({
                    'best_model/epoch': epoch,
                    'best_model/val_iou': val_metrics['iou'],
                    'best_model/val_dice': val_metrics['dice'],
                    'best_model/val_loss': val_loss
                })
                # Save best model to wandb
                wandb.save(str(save_path))

    # Test best model
    model.load_state_dict(torch.load(Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')['model_state_dict'])
    test_loss, test_metrics = validate(model, test_loader, criterion, device, n_classes, visualizer, epoch='final')

    if args.use_wandb:
        # Log test results
        wandb.log({
            'test/loss': test_loss,
            **{f'test/metrics/{k}': v for k, v in test_metrics.items()
               if k != 'confusion_matrix'},
            **{f'test/class_metrics/{k}': v for k, v in test_metrics.items()
               if k.startswith('class_')},
            'test/confusion_matrix': wandb.Image(
                plot_confusion_matrix(
                    test_metrics['confusion_matrix'],
                    title='Test Confusion Matrix'
                )
            )
        })

        # Log test visualizations
        test_viz_paths = list(Path('visualizations') / args.dataset / 'test').glob('*.png')
        wandb.log({
            'test/samples': [wandb.Image(str(p)) for p in test_viz_paths]
        })

        # Log final summary metrics
        wandb.run.summary.update({
            'best_val_iou': best_val_iou,
            'final_test_loss': test_loss,
            'final_test_iou': test_metrics['iou'],
            'final_test_dice': test_metrics['dice'],
            'total_epochs': args.epochs,
            'total_parameters': sum(p.numel() for p in model.parameters())
        })

        wandb.finish()

    print(f"\nTest Loss: {test_loss:.4f}, Test IoU: {test_metrics['iou']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training')

    parser.add_argument('--dataset', type=str, default='cityscapes',
                help='Dataset to use (kitti, cityscapes, pascal_voc, coco, or spacenet)')
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