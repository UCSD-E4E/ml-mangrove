import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Callable, List, Optional, Union, Tuple
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

def setup_device():
    """
    Sets up the device for training (GPU if available, else CPU).
    Returns:
        torch.device: The device to be used for training.
    """
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS) device.\n")
    else:
        DEVICE = torch.device("cpu")
        print("WARNING: No GPU found. Defaulting to CPU.")

    return DEVICE



class TrainingSession:
    """ Initializes the training session with the model, data loaders, loss function, and training parameters.
        Args:
            model (nn.Module): The model to be trained.
            trainLoader (DataLoader): DataLoader for the training dataset.
            testLoader (DataLoader): DataLoader for the test dataset.
            lossFunc (Callable): Loss function to be used for training.
            batch_size (int): Batch size for training.
            init_lr (float, optional): Initial learning rate for the optimizer. Defaults to 0.005.
            num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            device (torch.device, optional): Device to run the model on (CPU or GPU). If None, it will be set automatically.
            validation_dataset_names (List[str], optional): Names for the validation datasets. If None, datasets will be numbered.
            epoch_print_frequency (int, optional): Frequency of printing epoch information. Defaults to 1.
            optimizer (torch.optim.Optimizer, optional): Optimizer to use. If None, AdamW will be used.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. If None, CosineAnnealingLR will be used.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 1e-4.
            experiment_name (str, optional): Name for the experiment. If None, a timestamped name will be generated.
            save_checkpoints (bool, optional): Whether to save model checkpoints. Defaults to True.
            metrics_calculator (SegmentationMetrics, optional): Instance of SegmentationMetrics for calculating metrics.
        Attributes:
            training_loss (List[float]): List to store training loss for each epoch.
            validation_metrics (List[Dict[str, float]]): List to store validation metrics for each epoch.
            best_metric (float): Best metric value observed during training.
            best_epoch (int): Epoch number where the best metric was observed.
            experiment_dir (Path): Directory where experiment logs and checkpoints are saved.
            logger (logging.Logger): Logger for logging training progress and metrics.
            optimizer (torch.optim.Optimizer): Optimizer used for training.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler used during training.
            device (torch.device): Device on which the model is trained.
            model (nn.Module): The model being trained.
            multiple_test_loaders (bool): Indicates if multiple test loaders are used.
            scaler (GradScaler, optional): Gradient scaler for mixed precision training.
       """

    def __init__(self, model: nn.Module,
                 trainLoader: DataLoader,
                 testLoader: Union[DataLoader, List[DataLoader]], 
                 lossFunc: Callable, 
                 init_lr: float = 0.001, 
                 num_epochs: int = 10, 
                 device: Optional[torch.device] = None,
                 validation_dataset_names: Optional[List[str]] = None,
                 epoch_print_frequency: int = 1,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 weight_decay: float = 1e-4,
                 experiment_name: str = None, # type: ignore
                 save_checkpoints: bool = True):
        
        self.model = model.to(self.device)
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.lossFunc = lossFunc
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.device = setup_device() if device is None else device
        self.validation_dataset_names = validation_dataset_names
        self.epoch_print_frequency = epoch_print_frequency
        self.optimizer = optimizer if optimizer else AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=weight_decay)
        self.scheduler = scheduler if scheduler else optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        
        self.scaler = GradScaler(device=self.device.type) if self.device.type in ['cuda', 'cpu'] else None
        self.batch_size = trainLoader.batch_size if trainLoader.batch_size is not None else 1
        self.training_loss = []

        # Multiple test loaders handling
        if isinstance(testLoader, list):
            self.multiple_test_loaders = True
            self.validation_metrics = [[{} for _ in range(self.num_epochs)] for _ in range(len(testLoader))]
            if validation_dataset_names is not None and len(validation_dataset_names) != len(testLoader):
                print("WARNING: Number of validation dataset names does not match the number of test loaders.")
                self.validation_dataset_names = [f"Dataset {i + 1}" for i in range(len(testLoader))]
        else:
            self.multiple_test_loaders = False
            self.validation_metrics = self.num_epochs * [{}]
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0

        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path("experiments") / self.experiment_name
        if save_checkpoints:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self._setup_logging()
        
    
    def learn(self) -> Tuple[List[float], List[Union[dict, List[dict]]]]:
        """ Training with logging and checkpointing
        
            Returns:
                List[float]: training loss for each epoch,
                List[Union[dict, List[dict]]]: validation metrics for each epoch
        """

        if hasattr(self, 'logger'):
            self.logger.info(f"Starting training: {self.num_epochs} epochs")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        training_loss = []
        all_metrics = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            epoch_loss = self._train_epoch(epoch)
            training_loss.append(epoch_loss)
            
            # Validation phase
            if isinstance(self.testLoader, list):
                epoch_metrics = []
                for i, loader in enumerate(self.testLoader):
                    epoch_metrics.append(self.evaluate(loader))
                all_metrics.append(epoch_metrics)
                current_metric = epoch_metrics[-1].get('IOU', 0)
                is_best = current_metric > self.best_metric
            else:
                metrics = self.evaluate(self.testLoader)
                all_metrics.append(metrics)
                current_metric = metrics.get('IOU', 0)
                is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
            
            # Logging
            if (epoch + 1) % self.epoch_print_frequency == 0 or epoch == 0:
                self._log_epoch_results(epoch, epoch_loss, all_metrics[-1] if not isinstance(self.testLoader, list) else epoch_metrics)
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, all_metrics[-1] if not isinstance(self.testLoader, list) else epoch_metrics, is_best) # type: ignore
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(current_metric)
            else:
                self.scheduler.step()
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Training completed! Best metric: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
        
        return training_loss, all_metrics
    
    def _train_epoch(self, epoch):
        """Train single epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.trainLoader)
        
        pbar = tqdm(self.trainLoader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            
            # Forward pass
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                with autocast(device_type=self.device.type):
                    pred = self.model(x)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    loss = self.lossFunc(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                self.optimizer.step()
                loss = self.lossFunc(pred, y)
                loss.backward()
            if isinstance(pred, tuple):
                pred = pred[0]
            
            # Gather loss
            total_loss += loss.item()
            
            # Print progress
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """ evaluation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_metrics = {}
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            if self.scaler:
                with autocast(device_type=self.device.type):
                    pred = self.model(x)
            else:
                pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            metrics = self._calculate_segmentation_metrics(pred, y)
            loss = self.lossFunc(pred, y)
            total_loss += loss.item()
            
            for key, value in metrics.items():
                if key in all_metrics:
                    if key.startswith('class_'):
                        all_metrics[key] = [a + b for a, b in zip(all_metrics[key], value)]
                    else:
                        all_metrics[key] += value
                else:
                    all_metrics[key] = value

        # Average metrics
        for key in all_metrics:
            if key.startswith('class_'):
                all_metrics[key] = [v / len(dataloader) for v in all_metrics[key]]
            else:
                all_metrics[key] /= len(dataloader)
        
        all_metrics['Loss'] = total_loss / len(dataloader)
        return all_metrics

    def _calculate_segmentation_metrics(self, predictions, targets):
        """Calculate basic segmentation metrics"""
        # Ensure predictions always have shape [B, C, H, W]
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)  # [B, H, W] -> [B,1,H,W]

        if predictions.shape[1] == 1:
            # Binary segmentation -> sigmoid + threshold
            pred_classes = (torch.sigmoid(predictions) > 0.5).long().squeeze(1)  # [B,H,W]
        else:
            # Multi-class segmentation -> argmax
            pred_classes = torch.argmax(predictions, dim=1)  # [B,H,W]

        # Targets should end up as [B,H,W] with class indices
        if targets.dim() == 4:
            if targets.shape[1] == 1:
                # [B,1,H,W] -> [B,H,W]
                targets = targets.squeeze(1)
            else:
                # One-hot encoded [B,C,H,W] -> [B,H,W] class indices
                targets = torch.argmax(targets, dim=1)

        # Flatten tensors
        pred_flat = pred_classes.flatten()
        target_flat = targets.flatten()
        
        # Handle ignore index (common values: 255, -1)
        ignore_index = getattr(self.lossFunc, 'ignore_index', None)
        if ignore_index is not None and ignore_index != -100:  # -100 is default (no ignore)
            mask = target_flat != ignore_index
            pred_flat = pred_flat[mask]
            target_flat = target_flat[mask]
        
        
        # Overall pixel accuracy
        pixel_accuracy = (pred_flat == target_flat).float().mean().item()
        
        # Per-class metrics
        num_classes = predictions.shape[1]
        class_ious = np.zeros(num_classes)
        class_precisions = np.zeros(num_classes)
        class_recalls = np.zeros(num_classes)
        
        eps = 1e-8  # Avoid division by zero
        
        for class_id in range(num_classes):
            # Binary masks for current class
            pred_mask = (pred_flat == class_id)
            target_mask = (target_flat == class_id)
            
            # Calculate intersection and union
            tp = (pred_mask & target_mask).sum().float()
            fp = (pred_mask & ~target_mask).sum().float()
            fn = (~pred_mask & target_mask).sum().float()
            
            # Metrics for this class
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            iou = tp / (tp + fp + fn + eps)
            
            class_precisions[class_id] = precision.item()
            class_recalls[class_id] = recall.item()
            class_ious[class_id] = iou.item()
        
        # Aggregate metrics
        mean_precision = np.mean(class_precisions).item()
        mean_recall = np.mean(class_recalls).item()
        mean_iou = np.mean(class_ious).item()
        
        if len(class_ious) == 1:
            return {
                'Pixel_Accuracy': pixel_accuracy,
                'Precision': mean_precision,
                'Recall': mean_recall,
                'IOU': mean_iou,
            }
        
        return {
            'Pixel_Accuracy': pixel_accuracy,
            'Precision': mean_precision,
            'Recall': mean_recall,
            'IOU': mean_iou,
            'class_ious': class_ious,
            'class_precisions': class_precisions,
            'class_recalls': class_recalls
        }
    
    def _setup_logging(self):
        """Setup experiment logging"""
        log_file = self.experiment_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.experiment_name)
        
        # Save configuration
        config = {
            'model': str(self.model.__class__.__name__),
            'Loss Function': str(self.lossFunc.__class__.__name__),
            'optimizer': self.optimizer.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
            'learning_rate': self.init_lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'device': str(self.device)
        }
        
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        if not self.save_checkpoints:
            return
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'training_loss': self.training_loss,
            'validation_metrics': self.validation_metrics
        }
        
        # Save latest
        torch.save(checkpoint, self.experiment_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.experiment_dir / 'best.pth')
            if hasattr(self, 'logger'):
                self.logger.info(f"New best model saved at epoch {epoch}")
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(checkpoint, self.experiment_dir / f'epoch_{epoch}.pth')
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        torch.serialization.add_safe_globals([
            np.dtype,
            np.generic,
            np.float64,
            np.int64,
            np.dtypes.Float64DType,
            np._core.multiarray.scalar, # type: ignore
        ])

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def load_model_weights(self, path):
        """Load only model weights from file"""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

    def save_model(self, path):
        """Save only model weights to file"""
        torch.save(self.model.state_dict(), path)
    

    def _log_epoch_results(self, epoch, train_loss, val_metrics):
        """Enhanced logging of epoch results"""
        if isinstance(val_metrics, list):
            # Multiple datasets
            avg_loss = np.mean([m['Loss'] for m in val_metrics])
            avg_metric = np.mean([m.get('Mean_IoU', m.get('IOU', 0)) for m in val_metrics])
            
            log_msg = f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {avg_loss:.4f} | Avg Metric: {avg_metric:.4f}"
            print(log_msg)
            
            if hasattr(self, 'logger'):
                self.logger.info(log_msg)
                for i, metrics in enumerate(val_metrics):
                    dataset_name = self.validation_dataset_names[i] if self.validation_dataset_names else f"Dataset {i+1}"
                    self.logger.info(f"  {dataset_name}: {metrics}")
        else:
            # Single dataset
            val_loss = val_metrics['Loss']
            main_metric = val_metrics.get('Mean_IoU', val_metrics.get('IOU', 0))
            
            log_msg = f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {main_metric:.4f}"
            print(log_msg)
            
            if hasattr(self, 'logger'):
                self.logger.info(log_msg)
                self.logger.info(f"  Detailed metrics: {val_metrics}")
    
    def get_available_metrics(self) -> List[str]:
        """Returns a list of available metrics from the validation metrics."""
        if not self.validation_metrics or self.validation_metrics == []:
            print("No validation metrics available. Please run learn() first.")
            return []
        return list(self.validation_metrics[0].keys() if isinstance(self.validation_metrics[0], dict) else self.validation_metrics[0][0].keys())

    def get_metrics(self):
        """Returns the validation metrics for every epoch."""
        if not self.validation_metrics or self.validation_metrics == []:
            print("No validation metrics available. Please run learn() first.")
            return []
        return self.validation_metrics

    def plot_metrics(self, title, metrics=('Precision','Recall','IOU')):
        """Plot metrics with enhanced visualization"""
        if not self.validation_metrics:
            print("No metrics to plot. Run training first.")
            return
            
        plt.figure(figsize=(12, 8))
        
        if self.multiple_test_loaders:
            n_loaders = len(self.validation_metrics[0])
            names = (self.validation_dataset_names if self.validation_dataset_names and len(self.validation_dataset_names)==n_loaders
                    else [f"Dataset {i+1}" for i in range(n_loaders)])
            n_epochs = len(self.validation_metrics)
            xs = np.arange(1, n_epochs+1)

            for i in range(n_loaders):
                for m in metrics:
                    ys = [epoch_metrics[i].get(m, np.nan) for epoch_metrics in self.validation_metrics]
                    label = f"{names[i]} - {m}" if len(metrics) > 1 else f"{names[i]}"
                    plt.plot(xs, ys, label=label, marker='o', markersize=3)
        else:
            n_epochs = len(self.validation_metrics)
            xs = np.arange(1, n_epochs+1)
            for m in metrics:
                ys = [epoch_dict.get(m, np.nan) if isinstance(epoch_dict, dict) else np.nan for epoch_dict in self.validation_metrics]
                plt.plot(xs, ys, label=m, marker='o', markersize=3)

        plt.title(title, fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
        
        if hasattr(self, 'experiment_dir') and self.save_checkpoints:
            plt.savefig(self.experiment_dir / f'metrics_{title.replace(" ", "_").lower()}.png', 
                       dpi=150, bbox_inches='tight')
        plt.show()

    def plot_loss(self, title: str = "Training and Validation Loss", y_max: float = 0.3, training_time=None):
        """Enhanced loss plotting"""
        if not self.training_loss or not self.validation_metrics:
            raise ValueError("Training loss or validation metrics are not available. Please run learn() first.")
        
        plt.figure(figsize=(12, 8))

        # Training loss
        train_loss = [min(x, y_max) for x in self.training_loss]
        plt.plot(np.arange(1, len(train_loss)+1), train_loss, label="Train Loss", linewidth=2)
        
        # Validation loss
        if self.multiple_test_loaders:
            n_loaders = len(self.validation_metrics[0])
            names = (self.validation_dataset_names 
                    if self.validation_dataset_names and len(self.validation_dataset_names) == n_loaders
                    else [f"Dataset {i+1}" for i in range(n_loaders)])

            for i in range(n_loaders):
                validation_loss = [epoch_metrics[i]['Loss'] for epoch_metrics in self.validation_metrics]
                valid_loss = [min(x, y_max) for x in validation_loss]
                plt.plot(np.arange(1, len(valid_loss)+1), valid_loss,
                        label=f"{names[i]} - Validation Loss", linestyle='dashed', linewidth=2)
        else:
            validation_loss = [m['Loss'] for m in self.validation_metrics if isinstance(m, dict) and 'Loss' in m]
            valid_loss = [min(x, y_max) for x in validation_loss]
            plt.plot(np.arange(1, len(valid_loss)+1), valid_loss, label="Validation Loss", 
                    linestyle='dashed', linewidth=2)

        plt.title(title, fontsize=14)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        if training_time is not None:
            plt.text(0.02, 0.98, f"Training Time: {training_time}", transform=plt.gca().transAxes)

        step = y_max / 10
        yticks = np.arange(0, y_max+step, step)
        plt.yticks(yticks)

        xticks = np.arange(2, self.num_epochs+2, 2)
        plt.xticks(xticks)
        
        if hasattr(self, 'experiment_dir') and self.save_checkpoints:
            plt.savefig(self.experiment_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
        plt.show()

def calculate_class_weights(labels_path, classes, power=2.0):
    num_classes = len(classes)
    labels = np.load(labels_path, mmap_mode='r')
    unique, counts = np.unique(labels, return_counts=True)
    weights = np.ones(num_classes)
    total_pixels = counts.sum()

    print("Class distribution:")
    class_names = list(classes.values())
    for class_id, count in zip(unique, counts):
        if 0 <= class_id < len(class_names):
            pct = 100 * count / total_pixels
            print(f"  {class_names[class_id]:15}: {count:10,} pixels ({pct:5.1f}%)")
        
    for class_id, count in zip(unique, counts):
        if 0 <= class_id < num_classes:
            frequency = count / total_pixels
            # Use power to make weights more extreme
            weights[class_id] = (1.0 / frequency) ** power
    
    # Normalize so average weight is 1
    weights = weights / weights.mean()
    return torch.FloatTensor(weights)