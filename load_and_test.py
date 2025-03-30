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
import matplotlib.pyplot as plt
import random

from models.effisegnet import EffiSegNetBN

#%% Paths
images_path = "imagens_cancer_boca/images"
masks_path = "imagens_cancer_boca/masks"
model_path = "logs/efficientnet-b0_32/version_13/checkpoints/epoch=49-step=2350-v9.ckpt"
output_path = "results/predictions/v3"

#%% Simulated configuration object
class CFG:
    img_size = "derived"
    model = {
        "object": {
            "_target_": EffiSegNetBN, 
            "ch": 32,                               
            "pretrained": True,                       
            "freeze_encoder": False,              
            "deep_supervision": False,               
            "model_name": "efficientnet-b0"
        }
    }

cfg = CFG()         

#%% Get image size
if cfg.img_size == "derived":
    img_size = get_efficientnet_image_size(cfg.model['object']['model_name'])  
else:
    img_size = cfg.img_size

print(f"Image size: {img_size}")

#%% Define Albumentations transformation
test_transforms = A.Compose([
    A.Resize(*(img_size, img_size), interpolation=cv2.INTER_LANCZOS4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
    ToTensorV2()
])

#%% Load model checkpoint
# Assuming the model instantiation uses the 'instantiate' method (Hydra)
model = instantiate(cfg.model['object'])
checkpoint = torch.load(model_path)
state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

#%% Select images

def get_random_image_names(folder_path, num_images=10):
    # Lista todos os arquivos da pasta com extensões específicas
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
    if len(files) < num_images:
        raise ValueError("A pasta não contém imagens suficientes para selecionar a quantidade solicitada.")
    
    # Seleciona aleatoriamente `num_images` arquivos sem repetição
    selected_files = random.sample(files, num_images)
    
    # Remove as extensões dos nomes dos arquivos
    image_names = [os.path.splitext(f)[0] for f in selected_files]
    return image_names

test_indices = [
        20, 21, 48, 50, 54, 58, 71, 87, 99, 102, 106, 121, 130, 149, 151, 160, 169, 174, 187, 188, 189,
        191, 214, 235, 252, 257, 270, 276, 293, 308, 313, 330, 343, 344, 348, 359, 363, 372, 385, 411, 413
    ]

def get_specific_image_names(folder_path, test_indices):
    # Lista todos os arquivos da pasta com extensões específicas
    files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg','.JPG','.jpeg'))]
    
    # Ordena a lista de arquivos para garantir consistência nos índices
    files.sort()
    
    # Verifica se todos os índices fornecidos são válidos
    if max(test_indices) >= len(files):
        raise ValueError(
            f"Um ou mais índices estão fora do intervalo. "
            f"A pasta contém apenas {len(files)} imagens."
        )
    
    # Seleciona os arquivos correspondentes aos índices fornecidos
    selected_files = [files[i] for i in test_indices]
    
    # Remove as extensões dos nomes dos arquivos
    image_names = [os.path.splitext(f)[0] for f in selected_files]
    return image_names

# img_name_list = get_random_image_names(images_path, num_images=10)
img_name_list = get_specific_image_names(images_path, test_indices)
print("Nomes das imagens selecionadas:", img_name_list)

#%% Main Loop

for img_name in img_name_list:
    
    image_path = os.path.join(images_path,f"{img_name}.jpg")
    mask_path = os.path.join(masks_path,f"{img_name}.png")

    #%% Load the image
    image = cv2.imread(image_path)  # Read the image (BGR format)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    print(f"Original image shape: {image.shape}")
    
    # Apply the transformations
    transformed = test_transforms(image=image)
    test_image = transformed['image'].unsqueeze(0)  # Add batch dimension
    print(f"Transformed image shape (for model input): {test_image.shape}")
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {mask_path}")
    print(f"Original mask shape: {mask.shape}")

    #%% Prediction
    
    # Perform inference
    with torch.no_grad():
        output = model(test_image)
        predicted_mask = torch.sigmoid(output)
        predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension
        print(f"Output shape: {predicted_mask.shape}")
    
    # Postprocess the predicted mask (scale to 0-255)
    predicted_mask = (predicted_mask[0] * 255).astype(np.uint8)
    print("Predicted mask post-processed.")

    #%% Resultado visual
    
    # Ensure img_size is a tuple of integers
    if isinstance(img_size, int):
        img_size = (img_size, img_size)  # Convert to (width, height) format for square images
    
    # Redimensionar a imagem original e mascara para o tamanho desejado
    resized_image = cv2.resize(image, img_size, interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_LANCZOS4)
    
    # Converter saída inicial (antes de sigmoid) para numpy para visualização
    output_np = output.squeeze(0).squeeze(0).detach().cpu().numpy()  # Remover dimensão de batch
    
    # Normalizar a saída inicial para o intervalo [0, 1] para visualização
    output_normalized = (output_np - output_np.min()) / (output_np.max() - output_np.min())
    
    # Normalizar a máscara predita para o intervalo [0, 1] se necessário
    predicted_mask_normalized = predicted_mask / predicted_mask.max()

    # Expandir a dimensão da máscara para corresponder aos 3 canais da imagem
    predicted_mask_3c = predicted_mask_normalized[..., np.newaxis]

    # Aplicar a máscara à imagem preservando as cores originais
    segmented_region = resized_image * predicted_mask_3c


    # Configurar a figura para exibir as imagens
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(resized_image)
    # axes[0].set_title("Imagem Original")
    axes[0].axis("off")
    
    axes[1].imshow(resized_mask, cmap="gray")
    # axes[1].set_title("Máscara Original")
    axes[1].axis("off")
    
    axes[2].imshow(output_normalized, cmap="gray")  # Usar um mapa de cores para contraste
    # axes[2].set_title("Saída Inicial")
    axes[2].axis("off")
    
    axes[3].imshow(predicted_mask, cmap="gray")
    # axes[3].set_title("Máscara Predita")
    axes[3].axis("off")

    axes[4].imshow(segmented_region.astype("uint8"), cmap="gray")
    # axes[4].set_title("Exemplo visual")
    axes[4].axis("off")
    
    # Exibir ou salvar a imagem composta
    plt.tight_layout()
    # plt.show()
    
    result_img_path = os.path.join(output_path,f"{img_name}.png")
    plt.savefig(result_img_path, dpi=300)
    print(f"Prediction saved to {result_img_path}")
    plt.close()
    
