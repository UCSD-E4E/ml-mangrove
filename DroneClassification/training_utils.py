import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from models import *
from typing import List

class TrainingSession:
    def __init__(self, model: nn.Module, trainLoader: DataLoader, testLoader: DataLoader, lossFunc, INIT_LR=0.005, TRAIN_STEPS=100, NUM_EPOCHS=10):
        """ Initializes the training session with the model, data loaders, loss function, and training parameters.
        Args:
            model (nn.Module): The model to be trained.
            trainLoader (DataLoader): DataLoader for the training dataset.
            testLoader (DataLoader): DataLoader for the test dataset.
            lossFunc (Callable): Loss function to be used for training.
            num_epochs (int): Number of epochs to train the model.
            INIT_LR (float, optional): Initial learning rate for the optimizer. Defaults to 0.005.
            TRAIN_STEPS (int, optional): Number of training steps per epoch. Defaults to 100.
            NUM_EPOCHS (int, optional): Total number of epochs. Defaults to 10.
        """
        self.INIT_LR = INIT_LR
        self.TRAIN_STEPS = TRAIN_STEPS
        self.trainLoader = trainLoader
        self.testLoader = testLoader
        self.lossFunc = lossFunc
        self.num_epochs = NUM_EPOCHS
        self.training_loss = []
        self.validation_metrics = []
        self.DEVICE = setup_device()
        self.model = model.to(self.DEVICE)

    def learn(self):
        training_loss, validation_metrics = train(self.model, self.trainLoader, self.testLoader, self.lossFunc, self.DEVICE, self.INIT_LR, self.TRAIN_STEPS, self.num_epochs)
        self.training_loss = training_loss
        self.validation_metrics = validation_metrics

    def get_available_metrics(self)-> List[str]:
        """
        Returns the available metrics from the validation metrics.
        """
        if not self.validation_metrics:
            raise ValueError("Validation metrics are not available. Please run learn() first.")
        
        return list(self.validation_metrics[0].keys())

    def plot_metrics(self, title, metrics=['Precision', 'Recall', 'IOU'] ):
        """
        Plots the specified metrics over epochs.

        Args:
            title (str): Title of the plot.
            metric_dict (dict): Dictionary containing metrics for each epoch.
            metrics (List): List of metric names to plot. Defaults to ['Precision', 'Recall', 'IOU'].
        """
        plt.figure()
        for metric in metrics:
            plt.plot(np.arange(0, self.NUM_EPOCHS), [x[metric] for x in self.validation_metrics], label=metric)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="lower right")

        yticks = np.arange(0.0, 1.1, 0.1)
        plt.yticks(yticks)

        xticks = np.arange(2, self.num_epochs+2, 2)
        plt.xticks(xticks)
        
        plt.show()

    def plot_loss(self, title:str = "Training and Validation Loss", y_max:float = 0.3, training_time:str = None):
        """
        Plots the training and validation loss over epochs.
        Args:
            title (str): Title of the plot.
            y_max (float): Maximum value for the y-axis.
            training_time (str, optional): Time taken for training, displayed on the plot. Defaults to None.
        """

        if not self.training_loss or not self.validation_metrics:
            raise ValueError("Training loss or validation metrics are not available. Please run learn() first.")
        
        # scale losses to fit graph
        validation_loss = [x['Loss'] for x in self.validation_metrics]
        valid_loss = [min(x,y_max) for x in validation_loss]
        train_loss = [min(x, y_max) for x in self.training_loss]


        plt.figure()
        plt.plot(np.arange(1, self.num_epochs+1), train_loss, label="train_loss")
        plt.plot(np.arange(1, self.num_epochs+1), valid_loss, label="valid_loss")
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
def train(model, trainLoader : DataLoader, testLoader : DataLoader, lossFunc, DEVICE, INIT_LR, TRAIN_STEPS, NUM_EPOCHS, print_all_epochs = False):
  """
  Trains the model using the specified training and test loaders.
  Args:
    model (nn.Module): The model to be trained. 
    trainLoader (DataLoader): DataLoader for the training dataset.
    testLoader (DataLoader): DataLoader for the test dataset.
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

  for e in tqdm(range(NUM_EPOCHS)):
    # set the model in training mode
    model.train()
    totalTrainLoss = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
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
    metrics = evaluate(model, testLoader, lossFunc, DEVICE)
    all_metrics.append(metrics)
    avgTestLoss = metrics['Loss']

    if (e + 1) % 5 == 0 or e == 0 or print_all_epochs:
      # print the model training and validation information
      print("EPOCH: {}/{}".format(e + 1, NUM_EPOCHS)) #config
      print("Train loss: {:.6f}, Test loss: {:.4f}".format(
          avgTrainLoss, avgTestLoss))
      print("\nValidation Metrics:")
      for k, v in metrics.items():
          if k != 'Loss':
            print(f"{k}: {v}")
      print("\n")
  return training_loss, all_metrics

def evaluate(model: nn.Module,  dataloader: DataLoader, loss_func, DEVICE):
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