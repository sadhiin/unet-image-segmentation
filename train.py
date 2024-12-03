from asyncio import tasks
from sys import argv
from seaborn import load_dataset
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
    # Convert outputs to predictions
    predictions = torch.argmax(outputs, dim=1)

    # Initialize metrics dictionary
    metrics = {}

    # Calculate confusion matrix
    conf_matrix = torch.zeros((n_classes, n_classes), device=predictions.device)
    for t, p in zip(targets.view(-1), predictions.view(-1)):
        conf_matrix[t.long(), p.long()] += 1
    metrics['confusion_matrix'] = conf_matrix.cpu().numpy()

    # Calculate IoU and Dice for each class
    for class_id in range(n_classes):
        # Create binary masks for current class
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)

        # Calculate intersection and union
        intersection = torch.logical_and(pred_mask, target_mask).sum().item()
        union = torch.logical_or(pred_mask, target_mask).sum().item()

        # Calculate IoU
        iou = intersection / (union + 1e-10)
        metrics[f'class_{class_id}_iou'] = iou

        # Calculate Dice
        dice = 2 * intersection / (pred_mask.sum().item() + target_mask.sum().item() + 1e-10)
        metrics[f'class_{class_id}_dice'] = dice

    # Calculate mean metrics
    class_ious = [v for k, v in metrics.items() if k.endswith('_iou')]
    class_dices = [v for k, v in metrics.items() if k.endswith('_dice')]

    metrics['iou'] = np.mean(class_ious)
    metrics['dice'] = np.mean(class_dices)

    return metrics

def validate(model, val_loader, criterion, device, n_classes, visualizer, epoch):
    """
    Validate model
    Args:
        model: The neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on (cuda/cpu)
        n_classes: Number of classes
        visualizer: SegmentationVisualizer instance
        epoch: Current epoch number
    """
    model.eval()
    running_loss = 0.0
    all_metrics = []
    n_batches = len(val_loader)

    sample_images = None
    sample_masks = None
    sample_outputs = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            metrics = calculate_metrics(outputs, masks, n_classes)
            all_metrics.append(metrics)

            running_loss += loss.item()

            if batch_idx == 0:
                sample_images = images.detach().cpu()
                sample_masks = masks.detach().cpu()
                sample_outputs = outputs.detach().cpu()
                visualizer.visualize_batch(images, masks, outputs, phase='val', batch_idx=epoch)

                if args.use_wandb:
                    # Log sample predictions
                    wandb.log({
                        'val/samples': [wandb.Image(
                            visualizer.create_grid_image(
                                sample_images[i],
                                sample_masks[i],
                                sample_outputs[i],
                                f'Val Sample {i} - Epoch {epoch}'
                            )
                        ) for i in range(min(4, len(sample_images)))]
                    })

    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(m.get(metric, 0) for m in all_metrics) / n_batches

    return running_loss / n_batches, avg_metrics, (sample_images, sample_masks, sample_outputs)

def train_one_epoch(model, train_loader, criterion, optimizer, device, n_classes, visualizer, epoch):
    """
    Train model for one epoch
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        n_classes: Number of classes
        visualizer: SegmentationVisualizer instance
        epoch: Current epoch number
    """
    model.train()
    running_loss = 0.0
    all_metrics = []
    n_batches = len(train_loader)

    # Initialize progress bar
    pbar = tqdm(train_loader, desc='Training')

    # Store sample visualizations for logging
    sample_images = None
    sample_masks = None
    sample_outputs = None

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        metrics = calculate_metrics(outputs.detach(), masks.detach(), n_classes)
        all_metrics.append(metrics)

        # Update running loss
        running_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'iou': metrics['iou'],
            'dice': metrics['dice']
        })

        # Store first batch for visualization
        if batch_idx == 0:
            sample_images = images.detach().cpu()
            sample_masks = masks.detach().cpu()
            sample_outputs = outputs.detach().cpu()
            visualizer.visualize_batch(images, masks, outputs, phase='train', batch_idx=epoch)

            if args.use_wandb:
                # Log sample predictions
                wandb.log({
                    'train/samples': [wandb.Image(
                        visualizer.create_grid_image(
                            sample_images[i],
                            sample_masks[i],
                            sample_outputs[i],
                            f'Train Sample {i} - Epoch {epoch}'
                        )
                    ) for i in range(min(4, len(sample_images)))]
                })

    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(m.get(metric, 0) for m in all_metrics) / n_batches

    return running_loss / n_batches, avg_metrics, (sample_images, sample_masks, sample_outputs)

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

def test_model(model, test_loader, criterion, device, n_classes, visualizer):
    """Test the model and visualize results"""
    model.eval()
    running_loss = 0.0
    all_metrics = []
    n_batches = len(test_loader)

    sample_images = None
    sample_masks = None
    sample_outputs = None

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            metrics = calculate_metrics(outputs, masks, n_classes)
            all_metrics.append(metrics)

            running_loss += loss.item()

            if batch_idx == 0:
                sample_images = images.detach().cpu()
                sample_masks = masks.detach().cpu()
                sample_outputs = outputs.detach().cpu()
                visualizer.visualize_batch(images, masks, outputs, phase='test', batch_idx=0)

                if args.use_wandb:
                    # Log sample predictions
                    wandb.log({
                        'test/samples': [wandb.Image(
                            visualizer.create_grid_image(
                                sample_images[i],
                                sample_masks[i],
                                sample_outputs[i],
                                f'Test Sample {i}'
                            )
                        ) for i in range(min(4, len(sample_images)))]
                    })

    # Calculate average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = sum(m.get(metric, 0) for m in all_metrics) / n_batches

    return running_loss / n_batches, avg_metrics, (sample_images, sample_masks, sample_outputs)

def main(args):
    # Create base directories if they don't exist
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Setup device and print GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # Set CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        print("CUDA is not available. Using CPU.")

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

    print("Model architecture: \n")
    print(model)
    print("\n", "Device: ", device)
    if args.use_wandb:
        # Log model architecture
        wandb.watch(model, log='all', log_freq=100)

    # Loss function and optimizer

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Initialize visualizer
    visualizer = SegmentationVisualizer(args.dataset)

    # Training loop with mixed precision
    print("Starting training...")
    best_val_iou = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        try:
            # Train
            train_loss, train_metrics, train_samples = train_one_epoch(
                model, train_loader, criterion, optimizer, device, n_classes,
                visualizer, epoch  # Pass visualizer and epoch
            )

            # Validate
            val_loss, val_metrics, val_samples = validate(
                model, val_loader, criterion, device, n_classes,
                visualizer, epoch  # Pass visualizer and epoch
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
                       if k not in ['confusion_matrix']},  # Skip confusion matrix for regular metrics

                    # Validation metrics
                    **{f'val/metrics/{k}': v for k, v in val_metrics.items()
                       if k not in ['confusion_matrix']},  # Skip confusion matrix for regular metrics

                    # Per-class metrics
                    **{f'train/class_metrics/{k}': v for k, v in train_metrics.items()
                       if k.startswith('class_')},
                    **{f'val/class_metrics/{k}': v for k, v in val_metrics.items()
                       if k.startswith('class_')},

                    # Learning rate
                    'train/learning_rate': current_lr,
                })

                # Log confusion matrices separately
                if 'confusion_matrix' in train_metrics:
                    wandb.log({
                        'train/confusion_matrix': wandb.Image(
                            plot_confusion_matrix(
                                train_metrics['confusion_matrix'],
                                title=f'Train Confusion Matrix - Epoch {epoch}'
                            )
                        )
                    })

                if 'confusion_matrix' in val_metrics:
                    wandb.log({
                        'val/confusion_matrix': wandb.Image(
                            plot_confusion_matrix(
                                val_metrics['confusion_matrix'],
                                title=f'Val Confusion Matrix - Epoch {epoch}'
                            )
                        )
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

        except Exception as e:
            print(f"Error during training: {str(e)}")
            if args.use_wandb:
                wandb.finish()
            raise

    print("Training completed!")

    # Test best model
    print("\nTesting best model...")
    model.load_state_dict(torch.load(Path(args.checkpoint_dir) / f'{args.dataset}-best_model.pth')['model_state_dict'])
    test_loss, test_metrics, test_samples = test_model(
        model, test_loader, criterion, device, n_classes, visualizer
    )

    if args.use_wandb:
        # Log final test results
        wandb.log({
            'test/final_loss': test_loss,
            'test/final_iou': test_metrics['iou'],
            'test/final_dice': test_metrics['dice'],
            'test/confusion_matrix': wandb.Image(
                plot_confusion_matrix(
                    test_metrics['confusion_matrix'],
                    title='Final Test Confusion Matrix'
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main(args)