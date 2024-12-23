# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:41:31 2024
@author: vitor
"""

import torch    # Bib principal para computação em tensores
import hydra    # Configuração do experimento, gerenciando parâmetros em forma hierárquica
from hydra.utils import instantiate
from monai.networks.nets.efficientnet import get_efficientnet_image_size    # MONAI: bib para imagens médicas: obtem tamanho da imagem adequada para EfficientNet
import numpy as np  # Para conversão de arrays em listas
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


#%%

model_path = r"logs/efficientnet-b0_32/version_9/checkpoints/epoch=299-step=14100-v9.ckpt"
image_path = r"imagens_cancer_boca/images/FOP1182-18.jpg"
mask_path = r"imagens_cancer_boca/masks/FOP1182-18.png"

#%%

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    
    # Get image size
    if cfg.img_size == "derived":
        img_size = get_efficientnet_image_size(cfg.model.object.model_name)    # Obtem o tamanho adequado a partir do modelo se o tamanho for "derived"
    else:
        img_size = cfg.img_size     # Caso contrário, usa o tamanho especificado na configuração
    
    # Define Albumentations transformation
    test_transforms = A.Compose([
        A.Resize(*(img_size, img_size), interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        ToTensorV2()
    ])

    # Load the image
    image = cv2.imread(image_path)  # Read the image (BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Apply the transformations
    transformed = test_transforms(image=image)
    test_image = transformed['image'].unsqueeze(0)  # Add batch dimension

    # Load model checkpoint
    model = instantiate(cfg.model.object)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(test_image)
        predicted_mask = torch.sigmoid(output)
        predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension

    # Postprocess the predicted mask (scale to 0-255)
    predicted_mask = (predicted_mask[0] * 255).astype(np.uint8)

    # Save the predicted mask
    os.makedirs("results/predictions", exist_ok=True)
    output_path = "results/predictions/predicted_mask.png"
    cv2.imwrite(output_path, predicted_mask)
    print(f"Prediction saved to {output_path}")

if __name__ == "__main__":
    main()

#%% Load for Inference
# If you only need the model for testing or inference:
    
# model = Net(model=YourModel, criterion=YourCriterion, ...)  # Replace with your Net initialization
# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint["state_dict"])
# model.eval()  # Set the model to evaluation mode

#%% Load Entire Trainer
# If you want to resume training from this checkpoint:
    
# trainer = L.Trainer(resume_from_checkpoint="checkpoints/epoch=360-step=15162.ckpt")

#%% Testing the Loaded Model
# Once loaded, you can pass test images to the model:
    
# test_image = torch.tensor(your_image_data)  # Replace with your test image tensor
# test_image = test_image.unsqueeze(0)  # Add batch dimension if needed
# output = model(test_image)

#%% Verify Checkpoint Contents
# To inspect the contents of the checkpoint file:
    
# checkpoint = torch.load(model_path)
# print(checkpoint.keys())  # View all keys in the checkpoint

