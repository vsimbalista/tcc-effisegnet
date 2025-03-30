import os
import albumentations as A #bib para augmentação de imagens
import cv2
import lightning as L #treinamento de modelos em pytorch
from albumentations.pytorch import ToTensorV2 #converte img em tensor pytorch após augmentação
from torch.utils.data import DataLoader, Dataset, Subset #lidar com datasets, carregar, dividir em train/val

class KvasirSEGDatagen(Dataset):
    def __init__(self, pairs, transform=None):
        """
        Custom dataset class for loading image-mask pairs with optional transformations.
        
        Args:
            pairs (list of tuples): List of (image_path, mask_path) pairs.
            transform (albumentations.Compose, optional): Transformations to apply to the data.
        """
        self.transform = transform
        self.pairs = pairs

    def __len__(self):
        """Returns the total number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the image and mask at the given index.
        
        Args:
            idx (int): Index of the pair to retrieve.
        
        Returns:
            tuple: Transformed image tensor and mask tensor.
        """
        # Load the image in BGR format and convert to RGB.
        image = cv2.imread(self.pairs[idx][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load the mask in grayscale and binarize it.
        mask = cv2.imread(self.pairs[idx][1], 0)
        mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1]

        if self.transform is not None:
            # Apply the transformations if provided.
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Add a channel dimension to the mask and ensure it's a tensor.
        return image, mask.long().unsqueeze(0)

class KvasirSEGDataset(L.LightningDataModule):
    def __init__(self,
                 batch_size=64,
                 root_dir="./imagens_cancer_boca",
                 num_workers=os.cpu_count(),
                 img_size=(224, 224),
                 persistent_workers=True):
        """
        LightningDataModule for managing dataset splits and dataloaders.
        
        Args:
            batch_size (int): Number of samples per batch.
            root_dir (str): Root directory containing 'images' and 'masks' folders.
            num_workers (int): Number of subprocesses for data loading.
            img_size (tuple): Desired image size (height, width).
            persistent_workers (bool): Keep workers alive between epochs.
        """
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.img_size = img_size
        self.persistent_workers = persistent_workers

        # Indices for dataset splits.
        self.train_indices = None
        self.test_indices = None


    def get_transforms(self): #Está sendo usado nos 3 conjuntos.
        """Defines and returns transformations for testing."""
        return A.Compose([
            A.Resize(*(self.img_size, self.img_size), interpolation=cv2.INTER_LANCZOS4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
            ToTensorV2()
        ])

    def setup(self, stage=None):
        """
        Sets up the dataset by pairing image and mask files.
        
        Args:
            stage (str, optional): Stage of setup (e.g., 'fit', 'test').
        """
        # Get paths to all images and masks.
        images = os.listdir(os.path.join(self.root_dir, "images"))
        masks = os.listdir(os.path.join(self.root_dir, "masks"))
        images = [os.path.join(self.root_dir, "images", img) for img in images]
        masks = [os.path.join(self.root_dir, "masks", mask) for mask in masks]

        # Combine image and mask paths into pairs.
        all_pairs = list(zip(images, masks))
        
        # Initialize the dataset with testing transformations.
        self.dataset = KvasirSEGDatagen(all_pairs, transform=self.get_transforms())
    
    def set_splits(self, train_indices, test_indices):
        """
        Sets the indices for training, validation, and testing splits.
        
        Args:
            train_indices (list of int): Indices for training samples.
            val_indices (list of int): Indices for validation samples.
            test_indices (list of int): Indices for testing samples.
        """
        self.train_indices = train_indices
        self.test_indices = test_indices

    def train_dataloader(self):
        """Creates the training DataLoader."""
        train_subset = Subset(self.dataset, self.train_indices)
        return DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        """Creates the testing DataLoader."""
        test_subset = Subset(self.dataset, self.test_indices)
        return DataLoader(test_subset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers)