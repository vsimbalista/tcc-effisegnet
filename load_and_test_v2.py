# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:10:14 2024

@author: vitor
"""

import os
import cv2
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from hydra.utils import instantiate
from monai.networks.nets.efficientnet import get_efficientnet_image_size, EfficientNet    # MONAI: bib para imagens médicas: obtem tamanho da imagem adequada para EfficientNet


# Initialize the model
model = EfficientNet.from_name("efficientnet-b0")

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Adjust path
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Placeholder paths (update these as needed)
image_path = r"imagens_cancer_boca/images/FOP1182-18.jpg"
model_path = r"logs/efficientnet-b0_32/version_9/checkpoints/epoch=299-step=14100-v9.ckpt"

# Simulated configuration object
class CFG:
    img_size = "derived"
    model = {
        "object": {
            "model_name": "efficientnet-b0"
        }
    }

cfg = CFG()

# Get image size
if cfg.img_size == "derived":
    img_size = get_efficientnet_image_size(cfg.model["object"]["model_name"])  
else:
    img_size = cfg.img_size

print(f"Image size: {img_size}")

# Define Albumentations transformation
test_transforms = A.Compose([
    A.Resize(*(img_size, img_size), interpolation=cv2.INTER_LANCZOS4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
    ToTensorV2()
])

# Load the image
image = cv2.imread(image_path)  # Read the image (BGR format)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
print(f"Original image shape: {image.shape}")

# Apply the transformations
transformed = test_transforms(image=image)
test_image = transformed['image'].unsqueeze(0)  # Add batch dimension
print(f"Transformed image shape (for model input): {test_image.shape}")

# Load model checkpoint
# Assuming the model instantiation uses the 'instantiate' method (Hydra)
model = instantiate(cfg.model["object"])
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Model loaded successfully.")

# Perform inference
with torch.no_grad():
    output = model(test_image)
    predicted_mask = torch.sigmoid(output)
    predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension
    print(f"Output shape: {predicted_mask.shape}")

# Postprocess the predicted mask (scale to 0-255)
predicted_mask = (predicted_mask[0] * 255).astype(np.uint8)
print("Predicted mask post-processed.")

# Save the predicted mask
os.makedirs("results/predictions", exist_ok=True)
output_path = "results/predictions/predicted_mask.png"
cv2.imwrite(output_path, predicted_mask)
print(f"Prediction saved to {output_path}")
