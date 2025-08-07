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
    def __init__(self, model: nn.Module, trainLoader: DataLoader, testLoader: Union[DataLoader, List[DataLoader]], lossFunc, TRAIN_STEPS: int, INIT_LR: float = 0.005, NUM_EPOCHS: int = 10, device=None, validation_dataset_names: Optional[List[str]] = None):
        """ Initializes the training session with the model, data loaders, loss function, and training parameters.
        Args:
            model (nn.Module): The model to be trained.
            trainLoader (DataLoader): DataLoader for the training dataset.
            testLoader (DataLoader): DataLoader for the test dataset.
            lossFunc (Callable): Loss function to be used for training.
            TRAIN_STEPS (int): Number of training steps per epoch.
            INIT_LR (float, optional): Initial learning rate for the optimizer. Defaults to 0.005.
            NUM_EPOCHS (int, optional): Number of epochs to train the model. Defaults to 10.
            DEVICE (torch.device, optional): Device to run the model on (CPU or GPU). If None, it will be set automatically.
            validation_dataset_names (List[str], optional): Names for the validation datasets. If None, datasets will be numbered.
        """
        self.INIT_LR = INIT_LR
        self.TRAIN_STEPS = TRAIN_STEPS
        self.NUM_EPOCHS = NUM_EPOCHS
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.lossFunc = lossFunc
        self.training_loss = []
        self.epoch_print_frequency = 1  # Print metrics every x epochs
        self.validation_dataset_names = validation_dataset_names if validation_dataset_names is not None else None

        if isinstance(testLoader, list):
            self.multiple_test_loaders = True
            self.validation_metrics = [[{} for _ in range(self.NUM_EPOCHS)] for _ in range(len(testLoader))]
            if validation_dataset_names is not None and len(validation_dataset_names) != len(testLoader):
                print("WARNING: Number of validation dataset names does not match the number of test loaders. Reverting to numbering them as Dataset 1, Dataset 2, etc.")
                self.validation_dataset_names = [f"Dataset {i + 1}" for i in range(len(testLoader))]
        else:
            self.multiple_test_loaders = False
            self.validation_metrics = self.NUM_EPOCHS * [{}]


        if device is None:
            self.device = setup_device()
        else:
            self.device = device

        self.model = model.to(self.device)


    def learn(self):
        self.training_loss, self.validation_metrics = train(self.model, self.trainLoader, self.testLoader, self.lossFunc, self.device, self.INIT_LR, self.TRAIN_STEPS, self.NUM_EPOCHS,
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
    
    def set_training_parameters(self, INIT_LR: float, TRAIN_STEPS: int, NUM_EPOCHS: int):
        """ Sets the training parameters for the session.
        Args:
            INIT_LR (float): Initial learning rate for the optimizer.
            TRAIN_STEPS (int): Number of training steps per epoch.
            NUM_EPOCHS (int): Total number of epochs.
        """
        if INIT_LR <= 0 or TRAIN_STEPS <= 0 or NUM_EPOCHS <= 0:
            raise ValueError("INIT_LR, TRAIN_STEPS, and NUM_EPOCHS must be positive values.")
        
        self.INIT_LR = INIT_LR
        self.TRAIN_STEPS = TRAIN_STEPS
        self.NUM_EPOCHS = NUM_EPOCHS
    
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
            self.validation_metrics = [self.NUM_EPOCHS * [{}] for _ in range(len(testLoader))]

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
        Returns the validation metrics for every epoch.
        """
        if self.validation_metrics == []:
            print("No validation metrics available. Please run learn() first.")
            return []

        return self.validation_metrics

    def plot_metrics(self, title, metrics=['Precision', 'Recall', 'IOU'], validation_dataset_names: Optional[List[str]] = None):
        """
        Plots the specified metrics over epochs.

        Args:
            title (str): Title of the plot.
            metric_dict (dict): Dictionary containing metrics for each epoch.
            metrics (List): List of metric names to plot. Defaults to ['Precision', 'Recall', 'IOU'].
        """
        plt.figure()
        if self.multiple_test_loaders:
            # If multiple test loaders are used, plot metrics for each dataset
            for i, metrics_list in enumerate(self.validation_metrics):
                for metric in metrics:
                    plt.plot(np.arange(0, self.NUM_EPOCHS), [x[metric] for x in metrics_list], label=f"{validation_dataset_names[i] if validation_dataset_names else f'Dataset {i+1}'} - {metric}")
        else:
            for metric in metrics:
                plt.plot(np.arange(0, self.NUM_EPOCHS), [x[metric] for x in self.validation_metrics], label=metric)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="lower right")

        yticks = np.arange(0.0, 1.1, 0.1)
        plt.yticks(yticks)

        xticks = np.arange(2, self.NUM_EPOCHS+2, 2)
        plt.xticks(xticks)
        
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
        

        train_loss = [min(x, y_max) for x in self.training_loss]
        plt.plot(np.arange(1, self.NUM_EPOCHS+1), train_loss, label="train_loss")

        if self.multiple_test_loaders:
            # Plot each test loader's validation loss as a dotted line
            for i, metrics_list in enumerate(self.validation_metrics):  
                validation_loss = [x['Loss'] for x in metrics_list]
                valid_loss = [min(x, y_max) for x in validation_loss]
                plt.plot(np.arange(1, self.NUM_EPOCHS+1), valid_loss, label=f"{self.validation_dataset_names[i] if self.validation_dataset_names else f'Dataset {i+1}'} - Validation Loss", linestyle='dashed')
        else:
            # scale losses to fit graph
            metrics_list = self.validation_metrics
            validation_loss = [x['Loss'] for x in metrics_list if isinstance(x, dict) and 'Loss' in x]
            valid_loss = [min(x,y_max) for x in validation_loss]
            plt.plot(np.arange(1, self.NUM_EPOCHS+1), valid_loss, label="Validation Loss", linestyle='dashed')
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        if training_time is not None:
            plt.text(0, 0.3, f"Training Time: {training_time}")

        step = y_max / 10
        yticks = np.arange(0, y_max+step, step)  # Generate ticks from 0.025 to 0.3 with step 0.025
        plt.yticks(yticks)

        xticks = np.arange(2, self.NUM_EPOCHS+2, 2)  # Generate ticks from 0 to num_epochs with step 2
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
def train(model: nn.Module, trainLoader: DataLoader, testLoader: Union[DataLoader, List[DataLoader]], lossFunc: Callable, DEVICE: torch.device, INIT_LR: float, TRAIN_STEPS: int, NUM_EPOCHS: int, epoch_print_frequency: int = 5, validation_dataset_names: Optional[List[str]] = None):
    """
    Trains the model using the specified training and test loaders.
    Args:
      model (nn.Module): The model to be trained. 
      trainLoader (DataLoader): DataLoader for the training dataset.
      testLoader (DataLoader|List[DataLoader]): DataLoader(s) for the test dataset.
      lossFunc (Callable): Loss function to be used for training.
      DEVICE (torch.device): Device to run the model on (CPU or GPU).
      INIT_LR (float): Initial learning rate for the optimizer.
      TRAIN_STEPS (int): Number of training steps per epoch.
      NUM_EPOCHS (int): Number of epochs to train the model.
      print_all_epochs (bool): If True, prints training and validation information for all epochs. Else, prints only every 5 epochs.
    """
    opt = Adam(model.parameters(), lr=INIT_LR)
    # loop over epochs #config
    print("Learning...")
    training_loss = []
    all_metrics = []
    model.train()

    # loop over the number of epochs
    for e in tqdm(range(NUM_EPOCHS)):
        totalTrainLoss = 0
        # loop over the training set
        for (_, (x, y)) in enumerate(trainLoader):
            # send the input to the device
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()
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
        # calculate the average training
        avgTrainLoss = totalTrainLoss / TRAIN_STEPS
        training_loss.append(avgTrainLoss)

        # Evaluate on test dataset
        model.eval()
        if isinstance(testLoader, list):
            # If multiple test loaders are provided, evaluate on each one
            metrics = []
            test_losses = []    
            for loader in testLoader:
                metric = evaluate(model, loader, lossFunc, DEVICE)
                test_losses.append(metric['Loss'])
                metrics.append(metric)
            all_metrics.append(metrics)

            # print the model training and validation information
            print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS)) #config
            print("Train loss: {:.6f}, Average Test loss: {:.4f}".format(avgTrainLoss, np.mean(test_losses)))
            print("\nAll Test Datasets Metrics:")
            for i, metrics_list in enumerate(all_metrics):
                print(f"\n  Validation Metrics for {f'{validation_dataset_names[i] if validation_dataset_names else f'Dataset {i + 1}'}:'}")
                for metrics in metrics_list:
                    for k, v in metrics.items():
                        if k != 'Loss':
                            print(f"{k}: {v}")
            print("\n")
        
        else:
            metrics = evaluate(model, testLoader, lossFunc, DEVICE)
            all_metrics.append(metrics)
            avgTestLoss = metrics['Loss']

            if (e + 1) % epoch_print_frequency == 0 or e == 0:
                # print the model training and validation information
                print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
                print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))
                print("\nValidation Metrics:")
                for k, v in metrics.items():
                    if k != 'Loss':
                        print(f"{k}: {v}")
                print("\n")
                avgTestLoss = metrics['Loss']

    return training_loss, all_metrics

def evaluate(model: nn.Module,  dataloader: DataLoader, loss_func, DEVICE) -> dict:
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
            x = x.to(DEVICE)
            y = y.to(DEVICE).float()
            
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
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0
    specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0

    metrics = {
        'Landmass Captured': total_landmass_captured,
        'Loss': avg_loss,
        'Precision': precision,
        'Recall': recall,
        'f1_score': f1_score,
        'IOU': iou,
        'Accuracy': accuracy,
        'Specificity': specificity
    }

    return metrics



# Plotting functions


def plot_loss_comparison(title, training_losses1, training_losses2, validation_losses1, validation_losses2, NUM_EPOCHS, compare1 = "Satellite", compare2 = "ImageNet", y_max=0.3):
    # scale losses to fit graph
    valid_loss_sat = [min(x, y_max) for x in validation_losses1]
    train_loss_sat = [min(x, y_max) for x in training_losses1]
    valid_loss_img = [min(x, y_max) for x in validation_losses2]
    train_loss_img = [min(x, y_max) for x in training_losses2]
    
    plt.figure()
    plt.plot(np.arange(0, NUM_EPOCHS), train_loss_sat, label=f"Training loss {compare1}", color='orange')
    plt.plot(np.arange(0, NUM_EPOCHS), valid_loss_sat, label=f"Validation loss {compare1}", color='orange', linestyle='dashed')
    plt.plot(np.arange(0, NUM_EPOCHS), train_loss_img, label=f"Training loss {compare2}", color='teal')
    plt.plot(np.arange(0, NUM_EPOCHS), valid_loss_img, label=f"Validation loss {compare2}", color='teal', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    
    yticks = np.arange(0.025, 0.325, 0.025)  # Generate ticks from 0.025 to 0.3 with step 0.025
    plt.yticks(yticks)
    
    xticks = np.arange(2, NUM_EPOCHS+2, 2)  # Generate ticks from 0 to num_epochs with step 2
    plt.xticks(xticks)
    
    plt.show()


def plot_comparison_metrics(title, metrics: List[List[dict]], titles: List[str],
                             metrics_wanted = ['Precision', 'Recall', 'IOU'], x_label='Metrics', y_label = 'Values', y_lim = 1.1, 
                             size = (10.0, 6.0), single_metric=False):
    plt.figure(figsize=size)
    
    if single_metric:
        for i in range(len(titles)):
            plt.bar(titles[i], metrics[i][-1][metrics_wanted[0]])
    else:
        extracted_metrics = []
        for i in range(len(titles)):
            metrics_add = []
            for k in metrics[i][-1]:
                if k in metrics_wanted:
                    metrics_add.append(metrics[i][-1][k])
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
    
    plt.ylim(0, y_lim)
    plt.show()