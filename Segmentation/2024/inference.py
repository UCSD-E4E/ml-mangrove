
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from models import ResNet18_UNet

import cv2
import numpy as np
from matplotlib import pyplot as plt



def normalize(image):
    # Normalize the RGB image to the range [0, 1]
    return image / 255.0

def prediction(model, image, patch_size):
    # Initialize the segmented image with zeros
    segm_img = np.zeros(image.shape[:2])
    weights_sum = np.zeros(image.shape[:2])  # Initialize weights for normalization
    patch_num = 1
    
    # Iterate over the image in steps of patch_size
    for i in range(0, image.shape[0] - patch_size + 1, patch_size):   
        for j in range(0, image.shape[1] - patch_size + 1, patch_size): 
            # Extract the patch, ensuring we handle the boundaries
            single_patch = image[i:i+patch_size, j:j+patch_size]
            single_patch_norm = normalize(single_patch)
            single_patch_input = np.expand_dims(single_patch_norm, 0) 
            single_patch_input = np.transpose(single_patch_input, (0, 3, 1, 2))

            # Predict and apply Sigmoid
            with torch.no_grad():
                single_patch_input_tensor = torch.from_numpy(single_patch_input).float()
                output = model(single_patch_input_tensor)                
                print(output)
                threshold = 0
                binary_mask = (output > threshold).float()
                
                binary_mask_np = binary_mask.squeeze().detach().numpy()
                
              
            # Resize the prediction to match the patch size
            single_patch_prediction_resized = cv2.resize(binary_mask_np, (patch_size, patch_size))
            
            # Add the prediction to the segmented image and update weights for normalization
            segm_img[i:i+patch_size, j:j+patch_size] += single_patch_prediction_resized
            weights_sum[i:i+patch_size, j:j+patch_size] += 1
            
        
            patch_num += 1
            
            if patch_num % 100 == 0:
                print("Finished processing patch number", patch_num, "at position", i, j)
            if patch_num == 1000:
                return np.divide(segm_img, weights_sum, where=weights_sum > 0)
    # Normalize the final segmented image to handle overlaps
    segm_img = np.divide(segm_img, weights_sum, where=weights_sum > 0)

    return segm_img

patch_size = 256
# Load image and convert from BGR to RGB if needed
large_image = cv2.imread("/Users/gage/mangrove/data/jamaica3-31-34ortho-2-1.tif") 
large_image_rgb = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
model = ResNet18_UNet()

# Load the model weights
model.load_state_dict(torch.load('sat_resnet18_UNet_256_BCEweighted.pth', map_location=torch.device('mps')))

model.eval()

# Perform prediction
segmented_image = prediction(model, large_image_rgb, patch_size)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Large Image')
plt.imshow(large_image_rgb)  # RGB image for correct color display

# Create a custom colormap that maps grayscale values to yellow
yellow_cmap = mcolors.LinearSegmentedColormap.from_list('yellow_cmap', [(0, 'white'), (1, 'yellow')])

plt.subplot(122)
plt.title('Segmented Image')
plt.imshow(segmented_image, cmap=yellow_cmap)  # Use the custom colormap
plt.show()

# Save or visualize the segmented_image
cv2.imwrite('/Users/aaryanpanthi/Desktop/segmented_image.png', (segmented_image * 255).astype(np.uint8))
