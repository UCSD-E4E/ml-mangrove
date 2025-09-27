import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Callable, List, Optional, Union

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
            optimizer_type (str, optional): Type of optimizer to use ('adam', 'adamw', 'sgd'). Defaults to 'adamw'.
            scheduler_type (str, optional): Type of learning rate scheduler ('cosine', 'step', 'plateau'). Defaults to 'cosine'.
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
                 optimizer_type: str = 'adamw',
                 scheduler_type: str = 'cosine',
                 weight_decay: float = 1e-4,
                 experiment_name: str = None, # type: ignore
                 save_checkpoints: bool = True):
        
        # Device setup
        if device is None:
            self.device = self._setup_device()
        else:
            self.device = device
        self.model = model.to(self.device)

        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.batch_size = trainLoader.batch_size if trainLoader.batch_size is not None else 1
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.lossFunc = lossFunc
        self.training_loss = []
        self.epoch_print_frequency = epoch_print_frequency
        self.validation_dataset_names = validation_dataset_names
        
        # Enhanced features
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.weight_decay = weight_decay
        self.save_checkpoints = save_checkpoints
        
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

        # Enhanced optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Best model tracking
        self.best_metric = 0.0
        self.best_epoch = 0

        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path("experiments") / self.experiment_name
        if save_checkpoints:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self._setup_logging()
        
    def _setup_device(self):
        """device setup"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal Performance Shaders (MPS) device.")
        else:
            device = torch.device("cpu")
            print("WARNING: No GPU found. Using CPU.")
        return device
    
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
            'optimizer': self.optimizer_type,
            'scheduler': self.scheduler_type,
            'learning_rate': self.init_lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'device': str(self.device)
        }
        
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def _create_optimizer(self):
        """Create optimizer based on type"""
        if self.optimizer_type.lower() == 'adamw':
            return AdamW(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adam':
            return Adam(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.init_lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            return Adam(self.model.parameters(), lr=self.init_lr)
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.scheduler_type.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        elif self.scheduler_type.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=self.num_epochs//3, gamma=0.1)
        elif self.scheduler_type.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
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
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def learn(self):
        """Enhanced training loop"""
        self.training_loss, self.validation_metrics = self._train()
    
    def _train(self):
        """Enhanced training with better logging and checkpointing"""

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

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
                    epoch_metrics.append(self._evaluate(loader))
                all_metrics.append(epoch_metrics)
                
                # Check for best model
                avg_metric = np.mean([m.get('IOU', 0) for m in epoch_metrics])
                is_best = avg_metric > self.best_metric
                if is_best:
                    self.best_metric = avg_metric
                    self.best_epoch = epoch
                
            else:
                metrics = self._evaluate(self.testLoader)
                all_metrics.append(metrics)
                
                # Check for best model
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
                self.scheduler.step(current_metric if not isinstance(self.testLoader, list) else avg_metric)
            else:
                self.scheduler.step()
        
        if hasattr(self, 'logger'):
            self.logger.info(f"Training completed! Best metric: {self.best_metric:.4f} at epoch {self.best_epoch + 1}")
        
        return training_loss, all_metrics
    
    def _train_epoch(self, epoch):
        """Train single epoch with enhanced features"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.trainLoader)
        
        pbar = tqdm(self.trainLoader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            loss = self.lossFunc(pred, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _evaluate(self, dataloader):
        """ evaluation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_metrics = {}
        
        for x, y in dataloader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            loss = self.lossFunc(pred, y)
            total_loss += loss.item()
            metrics = self._calculate_segmentation_metrics(pred, y)

            for k, v in metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
    
        # Calculate comprehensive metrics
        for k, v in all_metrics.items():
            if k.startswith('class_'):
                metrics_list = np.mean(v, axis=0)
                all_metrics[k] = [round(float(item), 4) for item in metrics_list]
            else:
                all_metrics[k] = round(float(np.mean(v)), 4)

        all_metrics['Loss'] = round(total_loss / len(dataloader), 4)
        return all_metrics

    
    @torch.no_grad()
    def _calculate_segmentation_metrics(self, predictions, targets):
        """Calculate basic segmentation metrics"""
        
        # Get predicted classes
        pred_classes = torch.argmax(predictions, dim=1) if predictions.dim() == 4 else predictions
        
        # Squeeze targets
        if targets.dim() == 4:
            targets = targets.squeeze(1) if targets.shape[1] == 1 else torch.argmax(targets, dim=1)

        # Determine number of classes
        num_classes = predictions.shape[1] if predictions.dim() == 4 else max(int(targets.max()) + 1, int(pred_classes.max()) + 1)
        
        # Handle ignore index
        ignore_index = getattr(self.lossFunc, 'ignore_index', None)
        if ignore_index is not None and ignore_index != -100:
            valid_mask = targets != ignore_index
            pred_classes = pred_classes[valid_mask]
            targets = targets[valid_mask]
        
        # Flatten tensors
        pred_flat = pred_classes.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # Calculate pixel accuracy
        pixel_accuracy = (pred_flat == target_flat).float().mean().item()
        
        eps = 1e-8
        
        if num_classes > 1:
            # Vectorized per-class metrics using confusion matrix approach
            # Create one-hot encodings efficiently
            pred_onehot = torch.nn.functional.one_hot(pred_flat, num_classes).bool()
            target_onehot = torch.nn.functional.one_hot(target_flat, num_classes).bool()
            
            # Compute TP, FP, FN for all classes
            tp = (pred_onehot & target_onehot).sum(dim=0).float()
            fp = (pred_onehot & ~target_onehot).sum(dim=0).float()
            fn = (~pred_onehot & target_onehot).sum(dim=0).float()
            
            # Compute all metrics
            precisions = (tp / (tp + fp + eps)).cpu().numpy()
            recalls = (tp / (tp + fn + eps)).cpu().numpy()
            ious = (tp / (tp + fp + fn + eps)).cpu().numpy()
            
            return {
                'Pixel_Accuracy': pixel_accuracy,
                'Precision': float(precisions.mean()),
                'Recall': float(recalls.mean()),
                'IOU': float(ious.mean()),
                'class_ious': ious,
                'class_precisions': precisions,
                'class_recalls': recalls
            }
        else: # Binary case - calculate for the positive class (class 1)
            has_positive_pred = (pred_flat == 1).any()
            has_positive_target = (target_flat == 1).any()
            
            if not has_positive_pred and not has_positive_target:
                # No positive class in predictions or targets - perfect negative prediction
                precision = 1.0
                recall = 1.0
                iou = 1.0
            elif not has_positive_target:
                # No positive in target but model predicted positive - all false positives
                precision = 0.0
                recall = 1.0  # Undefined, but set to 1 (no positives to recall)
                iou = 0.0
            elif not has_positive_pred:
                # Target has positives but model predicted none - all false negatives
                precision = 1.0  # Undefined, but set to 1 (no predictions to be wrong)
                recall = 0.0
                iou = 0.0
            else:
                # Normal case - both classes present
                pred_mask = pred_flat == 1
                target_mask = target_flat == 1
                
                tp = (pred_mask & target_mask).sum().float()
                fp = (pred_mask & ~target_mask).sum().float()
                fn = (~pred_mask & target_mask).sum().float()
                
                precision = (tp / (tp + fp + eps)).item()
                recall = (tp / (tp + fn + eps)).item()
                iou = (tp / (tp + fp + fn + eps)).item()
            return {
                'Pixel_Accuracy': pixel_accuracy,
                'Precision': precision,
                'Recall': recall,
                'IOU': iou
            }

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