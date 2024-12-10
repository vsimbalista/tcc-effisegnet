import os
import albumentations as A #bib para augmentação de imagens
import cv2
import lightning as L #treinamento de modelos em pytorch
from albumentations.pytorch import ToTensorV2 #converte img em tensor pytorch após augmentação
from torch.utils.data import DataLoader, Dataset #lidar com datasets, carregar, dividir em train/val

class KvasirSEGDatagen(Dataset): # Generator em classe que herda da classe Dataset do Pytorch
    def __init__(self, pairs, transform=None):  # Inicialização da classe, com parametros
        self.transform = transform              # opções de augmentação aplicadas às imagens e máscaras
        self.pairs = pairs                      # recebe uma lista de pares de caminho (img, mascara)

    def __len__(self):                      # comprimento
        return len(self.pairs)              # retorna o tamanho do dataset (n de pares imagem-mascara)

    def __getitem__(self, idx):                                     # Acessa a imagem e a máscara no indice idx
        image = cv2.imread(self.pairs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # Converte imagem de BGR para RGB
        mask = cv2.imread(self.pairs[idx][1], 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]    # Máscara: leitura em cinza e binarizada em valor 127 (~1/2 de 255)

        if self.transform is not None:                              # transformações
            transformed = self.transform(image=image, mask=mask)    # aplica transformações na imagem e na máscara
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.long().unsqueeze(0)


class KvasirSEGDataset(L.LightningDataModule): # subclasse de lightiningdatamodule que organiza dados train/val/test
    def __init__(
        self,
        batch_size=64,              # batch size de treino
        # root_dir="./Kvasir-SEG",    # diretório das imagens original
        # root_dir="./imgsHae",   # NOVO DIRETORIO PARA TESTE
        root_dir="./imagens_cancer_boca",   # Diretorio final
        num_workers=os.cpu_count(), #original 2   # num de subprocessos para carregar dados (ideal que seja qtd de núcleos da CPU?)
        train_val_ratio=0.8,        # split de treino e validação
        img_size=(224, 224),        # tamanho da imagem para redimensionar
        persistent_workers=True,     # Mantém os workers ativos entre epochs para acelerar o carregamento
    ):
        super().__init__()                      # Super: acessa a classe pai ou superclasse: https://www.geeksforgeeks.org/python-super/
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.train_val_ratio = train_val_ratio
        self.img_size = img_size
        self.persistent_workers = persistent_workers

    def get_train_transforms(self):     # Transformações para TRAIN
        return A.Compose(               # agrupa várias transformações (redimentsionamento, flips h e v, ajuste de cor, elast, norm...)
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),    # Redimensionamento
                A.HorizontalFlip(p=0.5),                                                        # Flip horizontal
                A.VerticalFlip(p=0.5),                                                          # Flip vertical
                A.ColorJitter(                                                                  # Ajuste de cor
                    p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01
                ),
                A.Affine(                                                                       # Affine transformations (pode ter shear tbm mas aqui não usa)
                    p=0.5,
                    scale=(0.5, 1.5),                                                           # Scale (zoom in/out)
                    translate_percent=0.125,                                                    # Translação
                    rotate=90,                                                                  # Rotação
                    interpolation=cv2.INTER_LANCZOS4,
                ),
                A.ElasticTransform(p=0.5, interpolation=cv2.INTER_LANCZOS4),                    # Transformação elástica
                A.Normalize(                                                                    # Normalização
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),   # Converte imagem final para tensor pytorch
            ]
        )

    def get_val_transforms(self):     # Transformações para VAL (menos operações de augmentação)
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),    # Redimensionamento
                A.Normalize(                                                                    # Normalização
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),   # Converte imagem final para tensor pytorch
            ]
        )

    def get_test_transforms(self):     # Transformações para TEST (menos operações de augmentação)
        return A.Compose(
            [
                A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),    # Redimensionamento
                A.Normalize(                                                                    # Normalização
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255,
                ),
                ToTensorV2(),   # Converte imagem final para tensor pytorch
            ]
        )

    # Configuração do dataset
    def setup(self, stage=None):
        train_images = os.listdir(os.path.join(self.root_dir, "train/images"))                      # Leitura de PATH de imagens train
        train_masks = os.listdir(os.path.join(self.root_dir, "train/masks"))                        # Leitura de PATH de máscaras train
        train_images = [os.path.join(self.root_dir, "train/images", img) for img in train_images]   # Extração de PATH das imagens train
        train_masks = [os.path.join(self.root_dir, "train/masks", mask) for mask in train_masks]    # Extração de PATH das máscaras train

        val_images = os.listdir(os.path.join(self.root_dir, "validation/images"))
        val_masks = os.listdir(os.path.join(self.root_dir, "validation/masks"))
        val_images = [os.path.join(self.root_dir, "validation/images", img) for img in val_images]
        val_masks = [os.path.join(self.root_dir, "validation/masks", mask) for mask in val_masks]

        test_images = os.listdir(os.path.join(self.root_dir, "test/images"))
        test_masks = os.listdir(os.path.join(self.root_dir, "test/masks"))
        test_images = [os.path.join(self.root_dir, "test/images", img) for img in test_images]
        test_masks = [os.path.join(self.root_dir, "test/masks", mask) for mask in test_masks]

        train_pairs = list(zip(train_images, train_masks))  # Formando pares de PATH imagem-máscara em ZIP
        val_pairs = list(zip(val_images, val_masks))
        test_pairs = list(zip(test_images, test_masks))

        self.train_set = KvasirSEGDatagen(train_pairs, transform=self.get_train_transforms())   # Criando um generator para train, val e test sets
        self.val_set = KvasirSEGDatagen(val_pairs, transform=self.get_val_transforms())         # Passando dataset = pares de PATH | transform = método get_transforms para cada um
        self.test_set = KvasirSEGDatagen(test_pairs, transform=self.get_test_transforms())

    def train_dataloader(self):             # Dataloaders: dividem o dataset em lotes, preparam as imagens e aplicam as transformações
        return DataLoader(                  # DataLoader é uma função própria do pytorch
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,                   # shuffle TRUE para TRAIN
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers  # Adiciona o parâmetro persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,                  # shuffle FALSE para VAL e TEST
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers  # Adiciona o parâmetro persistent_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,                  # shuffle FALSE para VAL e TEST
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers  # Adiciona o parâmetro persistent_workers
        )