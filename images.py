# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:51:44 2024
@author: vitor
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
import os

def xml_to_mask(xml_path, image_path, save_path):
    """
    Converte um arquivo XML contendo vértices de regiões anotadas em uma máscara binária do tamanho da imagem original.
    
    Args:
        xml_path (str): Caminho para o arquivo XML.
        image_path (str): Caminho para a imagem original (JPG).
        save_path (str): Caminho para salvar a máscara binária gerada.
    """
    # Carregar a imagem para obter dimensões
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    height, width = image.shape[:2]

    # Criar máscara binária vazia
    mask = np.zeros((height, width), dtype=np.uint8)

    # Parse do arquivo XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterar sobre as regiões no XML
    for region in root.findall(".//Region"):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = int(float(vertex.attrib["X"]))
            y = int(float(vertex.attrib["Y"]))
            vertices.append((x, y))

        # Converter vértices para um array numpy e desenhar o polígono na máscara
        if vertices:
            points = np.array(vertices, dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)

    # Salvar a máscara
    cv2.imwrite(save_path, mask)
    print(f"Máscara salva em: {save_path}")

def process_folder(folder_path):
    """
    Processa todos os pares de arquivos XML e imagens JPG em um diretório.
    Para cada par, gera e salva uma máscara binária no formato PNG.
    
    Args:
        folder_path (str): Caminho para a pasta contendo os arquivos XML e imagens.
    """
    # Listar todos os arquivos na pasta
    files = os.listdir(folder_path)

    # Filtrar os arquivos XML e JPG
    xml_files = {f for f in files if f.endswith(".xml")}
    image_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}  # Suporte a diferentes extensões de imagens
    image_files = {f for f in files if os.path.splitext(f)[1] in image_extensions}


    # Processar pares XML e imagens com o mesmo nome base
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]  # Nome base do arquivo
        # Encontrar a imagem correspondente com qualquer extensão
        corresponding_image = next(
            (img for img in image_files if os.path.splitext(img)[0] == base_name), None
        )

        if corresponding_image:
            xml_path = os.path.join(folder_path, xml_file)
            image_path = os.path.join(folder_path, corresponding_image)
            save_path = os.path.join(folder_path, 'masks', f"{base_name}.png")

            # Converter o XML para máscara
            xml_to_mask(xml_path, image_path, save_path)
        else:
            print(f"Imagem correspondente não encontrada para: {xml_file}")

def find_images_without_masks(folder_path):
    """
    Verifica todas as imagens na pasta e identifica aquelas que não possuem 
    uma máscara correspondente (arquivo XML com o mesmo nome base).
    
    Args:
        folder_path (str): Caminho para a pasta contendo imagens e arquivos XML.
    """
    # Listar todos os arquivos na pasta
    files = os.listdir(folder_path)

    # Filtrar imagens e máscaras
    image_extensions = {".jpg", ".jpeg", ".JPG", ".JPEG"}  # Extensões válidas para imagens
    images = {os.path.splitext(f)[0] for f in files if os.path.splitext(f)[1] in image_extensions}
    masks = {os.path.splitext(f)[0] for f in files if f.endswith(".xml")}

    # Encontrar imagens sem máscaras correspondentes
    images_without_masks = images - masks

    # Exibir resultados
    if images_without_masks:
        print("Imagens sem máscara correspondente:")
        for image in images_without_masks:
            print(f"- {image}")
    else:
        print("Todas as imagens possuem máscara correspondente.")

# Exemplo de uso
folder_path = "imagens_cancer_boca"  # Substituir pelo caminho da pasta
process_folder(folder_path)
# find_images_without_masks(folder_path)

# plt.figure()
# plt.title('Máscara')
# plt.imshow(mask, cmap='gray')

