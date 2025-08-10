import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from models import *
from typing import Callable, List, Optional, Union

class TrainingSession:
    def __init__(self, model: nn.Module,
                    trainLoader: DataLoader,
                    testLoader: Union[DataLoader, List[DataLoader]], 
                    lossFunc: Callable, init_lr: float = 0.005, 
                    num_epochs: int = 10, 
                    device: Optional[torch.device] = None,
                    validation_dataset_names: Optional[List[str]] = None,
                    epoch_print_frequency: int = 1):
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
        """
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.batch_size = trainLoader.batch_size if trainLoader.batch_size is not None else 1
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.lossFunc = lossFunc
        self.training_loss = []
        self.epoch_print_frequency = epoch_print_frequency  # Print metrics every x epochs
        self.validation_dataset_names = validation_dataset_names if validation_dataset_names is not None else None

        if isinstance(testLoader, list):
            self.multiple_test_loaders = True
            self.validation_metrics = [[{} for _ in range(self.num_epochs)] for _ in range(len(testLoader))]
            if validation_dataset_names is not None and len(validation_dataset_names) != len(testLoader):
                print("WARNING: Number of validation dataset names does not match the number of test loaders. Reverting to numbering them as Dataset 1, Dataset 2, etc.")
                self.validation_dataset_names = [f"Dataset {i + 1}" for i in range(len(testLoader))]
        else:
            self.multiple_test_loaders = False
            self.validation_metrics = self.num_epochs * [{}]


        if device is None:
            self.device = setup_device()
        else:
            self.device = device

        self.model = model.to(self.device)

    def learn(self):
        self.training_loss, self.validation_metrics = train(self.model, self.trainLoader, self.testLoader, self.lossFunc, self.device, self.init_lr, self.num_epochs,
                                                             epoch_print_frequency=self.epoch_print_frequency, validation_dataset_names=self.validation_dataset_names)

    def set_epoch_print_frequency(self, frequency: int):
        """ Sets the frequency at which metrics are printed during training.
        Args:
            frequency (int): The frequency (in epochs) at which to print metrics.
        """
        if frequency <= 0:
            raise ValueError("Frequency must be a positive integer.")
        self.epoch_print_frequency = frequency
    
    def set_device(self, device: torch.device):
        """ Sets the device for training.
        Args:
            device (torch.device): The device to be used for training (CPU or GPU).
        """
        if not isinstance(device, torch.device):
            raise ValueError("Device must be a torch.device object.")
        self.device = device
        self.model.to(self.device)
    
    def set_loss_function(self, lossFunc: Callable):
        """ Sets the loss function for training.
        Args:
            lossFunc (Callable): The loss function to be used for training.
        """
        if not callable(lossFunc):
            raise ValueError("Loss function must be callable.")
        self.lossFunc = lossFunc

    def set_training_parameters(self, init_lr: float, train_steps: int, num_epochs: int):
        """ Sets the training parameters for the session.
        Args:
            init_lr (float): Initial learning rate for the optimizer.
            train_steps (int): Number of training steps per epoch.
            num_epochs (int): Total number of epochs.
        """
        if init_lr <= 0 or train_steps <= 0 or num_epochs <= 0:
            raise ValueError("init_lr, train_steps, and num_epochs must be positive values.")

        self.init_lr = init_lr
        self.train_steps = train_steps
        self.num_epochs = num_epochs

    def set_train_loader(self, trainLoader: DataLoader):
        """ Sets the training DataLoader.
        Args:
            trainLoader (DataLoader): DataLoader for the training dataset.
        """
        if not isinstance(trainLoader, DataLoader):
            raise ValueError("trainLoader must be an instance of torch.utils.data.DataLoader.")
        self.trainLoader = trainLoader

    def set_test_loader(self, testLoader: Union[DataLoader, List[DataLoader]]):
        """ Sets the test DataLoader.
        Args:
            testLoader (DataLoader|List[DataLoader]): DataLoader(s) for the test dataset.
        """
        if not isinstance(testLoader, (DataLoader, list)):
            raise ValueError("testLoader must be an instance of torch.utils.data.DataLoader or a list of DataLoaders.")

        if isinstance(testLoader, list):
            self.multiple_test_loaders = True
            if not all(isinstance(loader, DataLoader) for loader in testLoader):
                raise ValueError("All testLoaders in the list must be instances of torch.utils.data.DataLoader.")
            self.validation_metrics = [self.num_epochs * [{}] for _ in range(len(testLoader))]

        self.testLoader = testLoader
    
    def set_validation_dataset_names(self, validation_dataset_names: List[str]):
        """ Sets the names for the validation datasets.
        Args:
            validation_dataset_names (List[str]): List of names for the validation datasets.
        """
        if not isinstance(validation_dataset_names, list) or not all(isinstance(name, str) for name in validation_dataset_names):
            raise ValueError("validation_dataset_names must be a list of strings.")
        
        if self.multiple_test_loaders and len(validation_dataset_names) != len(self.testLoader):
            raise ValueError("Number of validation dataset names must match the number of test loaders.")
        
        self.validation_dataset_names = validation_dataset_names

    def get_available_metrics(self) -> List[str]:
        """
        Returns a list of available metrics from the validation metrics.
        """
        if self.validation_metrics == []:
            print("No validation metrics available. Please run learn() first.")
            return []

        return list(self.validation_metrics[0].keys() if isinstance(self.validation_metrics[0], dict) else self.validation_metrics[0][0].keys())

    def get_metrics(self):
        """
        Returns the validation metrics for every epoch in the form of a list of epochs of metrics.

        If there are multiple datasets, metrics are listed by epoch and then by dataset. A call of get_metrics()[0] will return the metrics of every dataset for the first epoch.
        A call of get_metrics()[0][0] will return metrics for the first dataset in the first epoch.

        If there is a single dataset, metrics are listed by epoch. A call of get_metrics()[0] will return the metrics for the first epoch.
        """
        if self.validation_metrics == []:
            print("No validation metrics available. Please run learn() first.")
            return []

        return self.validation_metrics

    def plot_metrics(self, title, metrics=('Precision','Recall','IOU')):
        plt.figure()

        if self.multiple_test_loaders:
            # transpose: collect per-dataset history across epochs
            n_loaders = len(self.validation_metrics[0])
            names = (self.validation_dataset_names if self.validation_dataset_names and len(self.validation_dataset_names)==n_loaders
                    else [f"Dataset {i+1}" for i in range(n_loaders)])
            n_epochs = len(self.validation_metrics)
            xs = np.arange(1, n_epochs+1)

            for i in range(n_loaders):
                # one line per (dataset, metric)
                for m in metrics:
                    ys = [epoch_metrics[i].get(m, np.nan) for epoch_metrics in self.validation_metrics]

                    if len(metrics) == 1:
                        label = f"{names[i]}"
                    else:
                        label = f"{names[i]} - {m}"
                    plt.plot(xs, ys, label=label)
        else:
            n_epochs = len(self.validation_metrics)
            xs = np.arange(1, n_epochs+1)
            for m in metrics:
                ys = [epoch_dict.get(m, np.nan) for epoch_dict in self.validation_metrics]
                plt.plot(xs, ys, label=m)

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="lower right")
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xticks(np.arange(2, n_epochs+2, 2))
        plt.show()

    def plot_loss(self, title:str = "Training and Validation Loss", y_max:float = 0.3, training_time = None):
        """
        Plots the training and validation loss over epochs.
        Args:
            title (str): Title of the plot.
            y_max (float): Maximum value for the y-axis.
            training_time (str, optional): Time taken for training, displayed on the plot. Defaults to None.
        """
        plt.figure()

        if not self.training_loss or not self.validation_metrics:
            raise ValueError("Training loss or validation metrics are not available. Please run learn() first.")
        
        # train_curve
        train_loss = [min(x, y_max) for x in self.training_loss]
        plt.plot(np.arange(1, len(train_loss)+1), train_loss, label="Train Loss")
        
        if self.multiple_test_loaders:
            # figure out number of loaders
            n_loaders = len(self.validation_metrics[0])
            names = (self.validation_dataset_names 
                    if self.validation_dataset_names and len(self.validation_dataset_names) == n_loaders
                    else [f"Dataset {i+1}" for i in range(n_loaders)])

            # plot each dataset across epochs
            for i in range(n_loaders):
                validation_loss = [epoch_metrics[i]['Loss'] for epoch_metrics in self.validation_metrics]
                valid_loss = [min(x, y_max) for x in validation_loss]
                plt.plot(np.arange(1, len(valid_loss)+1), valid_loss,
                        label=f"{names[i]} - Validation Loss", linestyle='dashed')
        else:
            metrics_list = self.validation_metrics  # list of dicts over epochs
            validation_loss = [m['Loss'] for m in metrics_list if isinstance(m, dict) and 'Loss' in m]
            valid_loss = [min(x, y_max) for x in validation_loss]
            plt.plot(np.arange(1, len(valid_loss)+1), valid_loss, label="Validation Loss", linestyle='dashed')

        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        if training_time is not None:
            plt.text(0, 0.3, f"Training Time: {training_time}")

        step = y_max / 10
        yticks = np.arange(0, y_max+step, step)  # Generate ticks from 0.025 to 0.3 with step 0.025
        plt.yticks(yticks)

        xticks = np.arange(2, self.num_epochs+2, 2)  # Generate ticks from 0 to num_epochs with step 2
        plt.xticks(xticks)
        
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

# Training Functions
def train(model: nn.Module, trainLoader: DataLoader, testLoader: Union[DataLoader, List[DataLoader]], lossFunc: Callable, device: torch.device, init_lr: float, num_epochs: int, epoch_print_frequency: int = 1, validation_dataset_names: Optional[List[str]] = None):
    """
    Trains the model using the specified training and test loaders.
    Args:
      model (nn.Module): The model to be trained. 
      trainLoader (DataLoader): DataLoader for the training dataset.
      testLoader (DataLoader|List[DataLoader]): DataLoader(s) for the test dataset.
      lossFunc (Callable): Loss function to be used for training.
      device (torch.device): Device to run the model on (CPU or GPU).
      init_lr (float): Initial learning rate for the optimizer.
      batch_size (int): Batch size for training.
      num_epochs (int): Number of epochs to train the model.
      epoch_print_frequency (int): Frequency of printing epoch information.
      validation_dataset_names (Optional[List[str]]): Names of the validation datasets.
    """
    model.to(device)
    opt = Adam(model.parameters(), lr=init_lr)
    training_loss = []
    all_metrics = []
    num_batches = 0
    
    print("Learning...")
    # loop over the number of epochs
    for e in range(num_epochs):
        print(f"Epoch {e + 1}/{num_epochs}:")
        model.train()
        totalTrainLoss = 0
        # loop over the training set
        for batch_idx, (x, y) in tqdm(enumerate(trainLoader), total=len(trainLoader), leave=True, mininterval=0.5, unit="batches"):
            # send the input to the device
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
            # perform a forward pass and calculate the training loss
            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            
            loss = lossFunc(pred, y)

            # first, zero out any previously accumulated gradients, then
            # perform backpropagation, and then update model parameters
            opt.zero_grad()
            loss.backward()
            opt.step()

            # add the loss to the total training loss so far
            totalTrainLoss += loss.item()
            num_batches += 1
        # calculate the average training
        avgTrainLoss = totalTrainLoss / max(num_batches, 1)
        training_loss.append(avgTrainLoss)

        # Evaluate on test dataset
        if isinstance(testLoader, list):
            names = (validation_dataset_names 
                    if validation_dataset_names and len(validation_dataset_names) == len(testLoader)
                    else [f"Dataset {i+1}" for i in range(len(testLoader))])

            epoch_metrics = []
            test_losses = []

            for loader in testLoader:
                m = evaluate(model, loader, lossFunc, device)
                epoch_metrics.append(m)
                test_losses.append(m['Loss'])

            all_metrics.append(epoch_metrics)

            print(f"Train loss: {avgTrainLoss:.6f}, Average Test loss: {np.mean(test_losses):.4f}")
            print("\nAll Test Datasets Metrics:")

            # print a labeled block for each dataset
            for name, m in zip(names, epoch_metrics):
                print(f"\n Validation Metrics for {name}:")
                for k, v in m.items():
                    print(f"  {k}: {v}")
            print()

            
        else:
            metrics = evaluate(model, testLoader, lossFunc, device)
            all_metrics.append(metrics)
            avgTestLoss = metrics['Loss']

            if (e + 1) % epoch_print_frequency == 0 or e == 0:
                # print the model training and validation information
                print("EPOCH: {}/{}".format(e + 1, num_epochs))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
                print("\nValidation Metrics:")
                for k, v in metrics.items():
                    if k != 'Loss':
                        print(f"{k}: {v}")
                print("\n")
                avgTestLoss = metrics['Loss']

    model.eval()
    return training_loss, all_metrics

def evaluate(model: nn.Module,  dataloader: DataLoader, loss_func, device: torch.device) -> dict:
    model.eval()
    total_loss = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0
    total_landmass_captured = 0
    total_landmass_actual = 0

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device).float()

            pred = model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = loss_func(pred, y)
            total_loss += loss.item()

            pred = torch.sigmoid(pred).view(-1)
            y = y.view(-1)
            
            TP = (pred * y).sum().item()
            FP = ((1 - y) * pred).sum().item()
            FN = (y * (1 - pred)).sum().item()
            TN = ((1 - y) * (1 - pred)).sum().item()
        
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

            total_landmass_actual += y.sum().item()
            total_landmass_captured += pred.sum().item()

    total_landmass_captured = total_landmass_captured / total_landmass_actual if total_landmass_actual > 0 else 0
    avg_loss = total_loss / len(dataloader)
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0
    specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0

    metrics = {
        'Landmass Captured': total_landmass_captured,
        'Loss': avg_loss,
        'Precision': precision,
        'Recall': recall,
        'IOU': iou,
        'Accuracy': accuracy,
        'Specificity': specificity
    }

    return metrics

# Model Comparison Plotting Functions
def plot_loss_comparison(title, training_losses1, training_losses2, validation_losses1, validation_losses2, num_epochs, compare1 = "Satellite", compare2 = "ImageNet", y_max=0.3):
    # scale losses to fit graph
    valid_loss_sat = [min(x, y_max) for x in validation_losses1]
    train_loss_sat = [min(x, y_max) for x in training_losses1]
    valid_loss_img = [min(x, y_max) for x in validation_losses2]
    train_loss_img = [min(x, y_max) for x in training_losses2]
    
    plt.figure()
    plt.plot(np.arange(0, num_epochs), train_loss_sat, label=f"Training loss {compare1}", color='orange')
    plt.plot(np.arange(0, num_epochs), valid_loss_sat, label=f"Validation loss {compare1}", color='orange', linestyle='dashed')
    plt.plot(np.arange(0, num_epochs), train_loss_img, label=f"Training loss {compare2}", color='teal')
    plt.plot(np.arange(0, num_epochs), valid_loss_img, label=f"Validation loss {compare2}", color='teal', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    
    yticks = np.arange(0.025, 0.325, 0.025)  # Generate ticks from 0.025 to 0.3 with step 0.025
    plt.yticks(yticks)
    
    xticks = np.arange(2, num_epochs+2, 2)  # Generate ticks from 0 to num_epochs with step 2
    plt.xticks(xticks)
    
    plt.show()

def plot_comparison_metrics(title, metrics: List[List[dict]], titles: List[str],
                             metrics_wanted = ['Precision', 'Recall', 'IOU'], x_label='Metrics', y_label = 'Values', y_lim = 1.1, 
                             size = (10.0, 6.0), single_metric=False):
    """
    Plots a comparison of metrics for different models.

    Args:
        title (str): The title of the plot.
        metrics (List[List[dict]]): The metrics to compare in the order of epoch, metrics
        titles (List[str]): The titles for set of metrics.
        metrics_wanted (List[str], optional): The specific metrics to plot. Defaults to ['Precision', 'Recall', 'IOU'].
        x_label (str, optional): The label for the x-axis. Defaults to 'Metrics'.
        y_label (str, optional): The label for the y-axis. Defaults to 'Values'.
        y_lim (float, optional): The limit for the y-axis. Defaults to 1.1.
        size (Tuple[float, float], optional): The size of the plot. Defaults to (10.0, 6.0).
        single_metric (bool, optional): Whether to plot a single metric or multiple metrics. Defaults to False.
    """
    plt.figure(figsize=size)
    
    if single_metric:
        for i in range(len(titles)):
            plt.bar(titles[i], metrics[-1][i][metrics_wanted[0]])
    else:
        extracted_metrics = []
        for i in range(len(titles)):
            metrics_add = []
            for k in metrics[-1][i]:
                if k in metrics_wanted:
                    metrics_add.append(metrics[-1][i][k])
            extracted_metrics.append(metrics_add)

        print(extracted_metrics)

        # Create bar positions
        bar_width = 0.8 / len(titles)  # Adjust bar width based on number of titles
        r = np.arange(len(metrics_wanted))
        
        for i in range(len(titles)):
            plt.bar([x + i * bar_width for x in r], extracted_metrics[i], width=bar_width, edgecolor='grey', label=titles[i])
        plt.xticks([r + bar_width * (len(titles) / 2) for r in range(len(metrics_wanted))], metrics_wanted)

        plt.legend()
    
    # Adding labels
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel(y_label, rotation=0, labelpad=len(y_label)*2)
    plt.title(title)

    if 'Landmass Captured' in metrics_wanted or 'Landmass Captured' in titles:
        plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Landmass Capture')

    plt.ylim(0, y_lim)
    plt.show()

def compare_series_by_resolution(title: str,
                                resolutions: list[str],
                                series_to_metrics: dict[str, list[dict]],
                                metric_keys: list[str],
                                y_label: str = "Value"
                                ):
    """
    Plot per-metric comparisons across resolutions for multiple series (e.g., Original vs MultiRes).

    Args:
        title: Title for the plot.
        resolutions: x-axis labels.
        series_to_metrics: mapping series_name -> list of metrics dicts per resolution, same order as `resolutions`.
                            Example: {
                                "Original": [dict_at_res0, dict_at_res1, ...],
                                "MultiRes": [dict_at_res0, dict_at_res1, ...],
                            }
        metric_keys: which metric names to plot (e.g., ["IOU"] or ["Precision","Recall","IOU"])
        y_label: y-axis label.
    """
    # sanity checks
    n = len(resolutions)
    for s, lst in series_to_metrics.items():
        assert len(lst) == n, f"Series '{s}' length {len(lst)} != resolutions length {n}"

    for mk in metric_keys:
        # build per-series maps: {resolution -> value_for_metric}
        metrics_for_plot = []
        series_names = list(series_to_metrics.keys())
        for s in series_names:
            vals = {resolutions[i]: series_to_metrics[s][i].get(mk) for i in range(n)}
            metrics_for_plot.append([vals])   # your plotter expects List[List[dict]] and uses [-1]

        plot_comparison_metrics(
            title,
            metrics_for_plot,
            titles=series_names,     # legend = series
            metrics_wanted=resolutions,  # x-axis = resolutions
            x_label="Resolutions",
            y_label=y_label,
            single_metric=False
        )