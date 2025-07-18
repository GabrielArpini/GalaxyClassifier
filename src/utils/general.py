import os
import numpy as np
from astroNN.datasets import load_galaxy10 
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

def get_project_root(marker: str = 'src'):
    """
    Get the root path of the project: ../GalaxyClassifier. Uses a marker to return the parent of the marker.

    Args:
    marker: A folder just bellow the root path.
    
    Returns:
    parent(Path): The root path of the project based on the marker definition.
    """
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker: {marker}")


def get_data(marker='src'):
    """
    Downloads data if it doesn't exist, if it exists loads it in variables images and labels
    and save them as .npy files in the data folder under the root path.

    Args:
    marker(str): name of a folder the folder which this function is called to call get_project_root.

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
    """
    Creates the Dataset with the .npy files with galaxy ndarray and label ndarray.
    It uses a lazy approach because my RAM (16 GB) can't load it all when splitting data and runing model.
    It uses the indices and mmap_mode to read the data without loading the entire data with the DataLoaders.
    """
    def __init__(self, indices, image_path, label_path, transform=None):
        self.indices = indices
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.transform = transform
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image = self.images[actual_idx]
        label = self.labels[actual_idx]

        label = torch.tensor(label, dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
        return image, label



if __name__ == '__main__':
    get_data()
