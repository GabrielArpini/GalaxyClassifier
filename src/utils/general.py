import os
import numpy as np
from astroNN.datasets import load_galaxy10 
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

def get_project_root(marker: str = 'src'):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker: {marker}")


def get_data(marker='src'):
    """
    Downloads data if it doesn't exist, if it exists loads it in variables images and labels

    Args:
    marker(str): name of a file to get the path to.

    Returns:
    images(ndarray): A numpy array with the images.
    labels(ndarray): A numpy array with the labels.
    """
    root_path = get_project_root(marker)
    path = root_path / "data"
    images_path = os.path.join(path, 'images.npy')
    labels_path = os.path.join(path, 'labels.npy')
    
    if os.path.exists(images_path) and os.path.exists(labels_path):
        images = np.load(images_path)
        labels = np.load(labels_path)
    else:
        images, labels = load_galaxy10()
        os.makedirs(path, exist_ok=True)
        np.save(images_path, images)
        np.save(labels_path, labels)
    
    return images, labels


class LazyGalaxyDataset(Dataset):
    def __init__(self, indices, image_path, label_path, transform=None):
        self.indices = indices
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.transform = transform
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.images[actual_idx] / 255.0
        label = self.labels[actual_idx]

        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.int64)
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor, label_tensor


def get_dataset(train_indices, test_indices, val_indices, images_path, labels_path):
    '''
    Gets the dataset for training, testing, and validation.
    Args:
        train_indices (np.ndarray): Actual indices for training samples.
        test_indices (np.ndarray): Actual indices for testing samples.
        val_indices (np.ndarray): Actual indices for validation samples.
    Returns:
        tuple: Three datasets for training, testing, and validation.
    '''
    
    train_dataset = LazyGalaxyDataset(train_indices, images_path, labels_path)
    test_dataset = LazyGalaxyDataset(test_indices, images_path, labels_path)
    valid_dataset = LazyGalaxyDataset(val_indices, images_path, labels_path)
    
    return train_dataset, test_dataset, valid_dataset

if __name__ == '__main__':
    get_data()
