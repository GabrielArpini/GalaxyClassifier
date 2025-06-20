import os
import numpy as np
from astroNN.datasets import load_galaxy10 
import torch
from torch.utils.data import Dataset
import numpy as np


def get_data(path='../../data'):
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
    def __init__(self, indices, image_path, label_path):
        self.indices = indices
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.images[actual_idx] / 255.0
        label = self.labels[actual_idx]

        image_tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.int64)

        return image_tensor, label_tensor


def get_dataset(X_train,X_test,X_val):
    '''
    Gets the dataset for training, testing, and validation.
    Args:
        X_train (np.ndarray): Training images.
        X_test (np.ndarray): Testing images.
        X_val (np.ndarray): Validation images.
    Returns:
        tuple: Three datasets for training, testing, and validation.
    '''
    
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))
    valid_indices = np.arange(len(X_val))

    images_path = '/content/drive/MyDrive/GalaxyDataset/images.npy' # Insert the correct path
    labels_path = '/content/drive/MyDrive/GalaxyDataset/labels.npy' # Insert the correct path

    train_dataset = LazyGalaxyDataset(train_indices, images_path, labels_path)
    test_dataset = LazyGalaxyDataset(test_indices, images_path, labels_path)
    valid_dataset = LazyGalaxyDataset(valid_indices, images_path, labels_path)
    return train_dataset, test_dataset, valid_dataset
