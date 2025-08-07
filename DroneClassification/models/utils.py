from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from typing import List

# Training Functions
def train(model, trainLoader : DataLoader, testLoader : DataLoader, lossFunc, TRAIN_STEPS, init_lr=0.005, num_epochs=10, print_all_epochs = False, DEVICE=torch.device("cpu")):
  opt = Adam(model.parameters(), lr=init_lr)
  # loop over epochs #config
  print("[INFO] training the network...")
  training_loss = []
  all_metrics = []

  for e in tqdm(range(num_epochs)):
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
    metrics = evaluate(model, testLoader, lossFunc, DEVICE=DEVICE)
    all_metrics.append(metrics)
    avgTestLoss = metrics['Loss']

    if (e + 1) % 5 == 0 or e == 0 or print_all_epochs:
      # print the model training and validation information
      print("EPOCH: {}/{}".format(e + 1, num_epochs)) #config
      print("Train loss: {:.6f}, Test loss: {:.4f}".format(
          avgTrainLoss, avgTestLoss))
      print("\nValidation Metrics:")
      for k, v in metrics.items():
          if k != 'Loss':
            print(f"{k}: {v}")
      print("\n")
  return training_loss, all_metrics

def evaluate(model: nn.Module, dataloader: DataLoader, loss_func, DEVICE=torch.device("cpu")) -> Dict[str, float]:
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


# Plotting Functions
def plot_losses(title, training_loss, validation_loss, training_time=None, y_max=0.3):
  """
  Plot training and validation losses over epochs.

Args:
    title (str): Title of the plot.
    training_loss (List[float]): List of training losses for each epoch.
    validation_loss (List[float]): List of validation losses for each epoch.
    training_time (str, optional): Training time to display on the plot. Defaults to None.
    y_max (float): Maximum value for y-axis scaling. Defaults to 0.3.
  """

  total_epochs = len(training_loss)
  if total_epochs == 0:   
      print("No losses to plot.")
      return
  
  # scale losses to fit graph
  valid_loss = [min(x,y_max) for x in validation_loss]
  train_loss = [min(x, y_max) for x in training_loss]


  plt.figure()
  plt.plot(np.arange(1, total_epochs+1), train_loss, label="train_loss")
  plt.plot(np.arange(1, total_epochs+1), valid_loss, label="valid_loss")
  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc="upper right")
  if training_time is not None:
    plt.text(0, 0.3, f"Training Time: {training_time}")

  step = y_max / 10
  yticks = np.arange(0, y_max+step, step)  # Generate ticks from 0.025 to 0.3 with step 0.025
  plt.yticks(yticks)

  xticks = np.arange(2, total_epochs+2, 2)  # Generate ticks from 0 to num_epochs with step 2
  plt.xticks(xticks)
  
  plt.show()

def plot_metrics(title: str, metric_dict: Dict, metrics: List = ['Precision', 'Recall', 'IOU']):
    """
    Plot metrics over epochs.
    Args:
        title (str): Title of the plot.
        metric_dict (Dict): Dictionary containing metrics for each epoch.
        metrics (List[str]): List of metrics to plot. Defaults to ['Precision', 'Recall', 'IOU'].
    """
    total_epochs = len(metric_dict)
    if total_epochs == 0:   
        print("No metrics to plot.")
        return
    
    plt.figure()
    for metric in metrics:
        plt.plot(np.arange(0, total_epochs), [x[metric] for x in metric_dict], label=metric)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend(loc="lower right")

    yticks = np.arange(0.0, 1.1, 0.1)
    plt.yticks(yticks)

    xticks = np.arange(2, total_epochs+2, 2)
    plt.xticks(xticks)
    
    plt.show()

def plot_loss_comparison(title, training_losses1, training_losses2, validation_losses1, validation_losses2, compare1 = "Satellite", compare2 = "ImageNet", y_max=0.3):
    """ 
    Compare training and validation losses of two training sessions and plot them.

    Args:
        title (str): Title of the plot.
        training_losses1 (List[float]): Training losses for the first model.
        training_losses2 (List[float]): Training losses for the second model.
        validation_losses1 (List[float]): Validation losses for the first model.
        validation_losses2 (List[float]): Validation losses for the second model.
        compare1 (str): Name of the first model for comparison.
        compare2 (str): Name of the second model for comparison.
        y_max (float): Maximum value for y-axis scaling.
    """
    num_epochs = len(training_losses1)
    if num_epochs != len(training_losses2) or num_epochs != len(validation_losses1) or num_epochs != len(validation_losses2):
        raise ValueError("All loss lists must have the same length.")

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

def plot_metrics_comparison(title, metrics: List[List[Dict]], titles: List[str], metrics_wanted = ['Precision', 'Recall', 'IOU']):
    """
    Compare final evaluation metrics and plot them in a bar chart.
    
    Args:
        title (str): Title of the plot.
        metrics (List[List[Dict]]): List of metrics for each model, where each model's metrics is a list of dictionaries.
        titles (List[str]): List of titles for each model.
        metrics_wanted (List[str]): List of metrics to extract and plot.
        
    """
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
    
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    
    for i in range(len(titles)):
        plt.bar([x + i * bar_width for x in r], extracted_metrics[i], width=bar_width, edgecolor='grey', label=titles[i])
    
    # Adding labels
    plt.xlabel('Metrics', fontweight='bold')
    plt.ylabel('Values')
    plt.title(title)
    plt.xticks([r + bar_width * (len(titles) / 2) for r in range(len(metrics_wanted))], metrics_wanted)
    plt.ylim(0, 1.1)
    
    plt.legend()
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