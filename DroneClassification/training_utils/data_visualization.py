import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List


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

def visualize_segmentation_results(model, dataset, raw_images, sample_idx=0, class_names=None, device: torch.device = torch.device('cpu')):
    """
    Properly visualize segmentation results with denormalized images
    """
    # Get sample data
    sample_image, sample_target = dataset[sample_idx]

    print(f"Sample {sample_idx} info:")
    print(f"  Image shape: {sample_image.shape}, range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    print(f"  Target shape: {sample_target.shape}, unique values: {torch.unique(sample_target)}")
    
    # Get model prediction
    model.eval().to(device)
    with torch.no_grad():
        if sample_image.dim() == 3:
            model_input = sample_image.unsqueeze(0).to(device)  # Add batch dimension
        else:
            model_input = sample_image.to(device)
            
        # Forward pass
        logits = model(model_input)
        if isinstance(logits, tuple):   # some models return (out, aux)
            logits = logits[0]
        
        # pick most likely class per pixel
        prediction = torch.argmax(logits.squeeze(0), dim=0)  # [H,W]


    print(f"  Prediction shape: {prediction.shape}, range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    # Bring single image into memory if it's a memmap
    img = raw_images[sample_idx].copy()
    img = np.transpose(img, (1, 2, 0))
    img_display = img
    

    # 2. Prepare target for display
    target_display = sample_target.squeeze().cpu().numpy()
    
    # 3. Prepare prediction for display
    pred_display = prediction.cpu().numpy()
    
    # 4. Create difference map
    if target_display.shape == pred_display.shape:
        diff_map = np.abs(target_display - pred_display)
    else:
        diff_map = None
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth
    im1 = axes[0, 1].imshow(target_display, cmap='tab10', vmin=0, vmax=5)
    axes[0, 1].set_title(f'Ground Truth \n Classes: {np.unique(target_display)}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Prediction
    im2 = axes[0, 2].imshow(prediction.cpu().numpy(), cmap='tab10', vmin=0, vmax=5)
    axes[0, 2].set_title("Prediction (class IDs)")
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Difference map
    if diff_map is not None:
        im3 = axes[1, 0].imshow(diff_map, cmap='Reds')
        axes[1, 0].set_title(f'Absolute Difference \n Mean: {diff_map.mean():.4f}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    else:
        axes[1, 0].text(0.5, 0.5, 'Shape Mismatch', ha='center', va='center', 
                       transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('Cannot compute difference')
        axes[1, 0].axis('off')
    
    # Class distribution in target
    unique, counts = np.unique(target_display, return_counts=True)
    axes[1, 1].bar(unique, counts)
    axes[1, 1].set_title('Ground Truth Class Distribution')
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Pixel Count')
    
    # Add class names if provided
    if class_names and len(class_names) >= len(unique):
        axes[1, 1].set_xticks(unique)
        axes[1, 1].set_xticklabels([class_names[int(i)] for i in unique], rotation=45)
    
    # Class distribution in prediction
    if pred_display.dtype.kind in ['i', 'u']:  # Integer predictions
        pred_rounded = pred_display.astype(int)
    else:  # Float predictions - round to nearest integer
        pred_rounded = np.round(pred_display).astype(int)
    
    unique_pred, counts_pred = np.unique(pred_rounded, return_counts=True)
    axes[1, 2].bar(unique_pred, counts_pred, alpha=0.7)
    axes[1, 2].set_title('Prediction Class Distribution')
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Pixel Count')
    
    if class_names and len(class_names) >= len(unique_pred):
        axes[1, 2].set_xticks(unique_pred)
        axes[1, 2].set_xticklabels([class_names[int(i)] for i in unique_pred], rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS:")
    print("="*60)
    
    print(f"Target class distribution:")
    total_pixels = target_display.size
    for class_val, count in zip(unique, counts):
        percentage = 100 * count / total_pixels
        class_name = class_names[int(class_val)] if class_names else f"Class {class_val}"
        print(f"  {class_name}: {count:,} pixels ({percentage:.1f}%)")
    
    print(f"\nPrediction statistics:")
    print(f"  Raw prediction range: [{pred_display.min():.4f}, {pred_display.max():.4f}]")
    print(f"  Raw prediction mean: {pred_display.mean():.4f}")
    
    if diff_map is not None:
        print(f"\nAccuracy metrics:")
        exact_matches = (diff_map == 0).sum()
        accuracy = exact_matches / total_pixels
        print(f"  Pixel-wise accuracy: {accuracy:.4f} ({100*accuracy:.2f}%)")
        print(f"  Mean absolute error: {diff_map.mean():.4f}")
        print(f"  Max error: {diff_map.max():.4f}")
        
        if accuracy == 1.0:
            print("  🎉 Perfect predictions! But check if this is realistic...")
        elif accuracy > 0.9:
            print("  ✅ Very good predictions!")
        elif accuracy > 0.7:
            print("  📊 Decent predictions")
        else:
            print("  ⚠️  Poor predictions - model needs more training")
