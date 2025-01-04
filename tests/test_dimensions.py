import os
from PIL import Image

def find_image_dimensions(folder_path):
    min_dimension = float('inf')
    max_dimension = 0
    min_image = ""
    max_image = ""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    smallest = min(width, height)
                    largest = max(width, height)

                    if smallest < min_dimension:
                        min_dimension = smallest
                        min_image = filename
                    if largest > max_dimension:
                        max_dimension = largest
                        max_image = filename

            except Exception as e:
                print(f"Erro ao processar {filename}: {e}")

    return min_dimension, max_dimension, min_image, max_image

# Exemplo de uso
folder_path = "imagens_cancer_boca/images"
min_dim, max_dim, min_img, max_img = find_image_dimensions(folder_path)

print(f"Menor dimensão encontrada: {min_dim}px (Imagem: {min_img})")
print(f"Maior dimensão encontrada: {max_dim}px (Imagem: {max_img})")
