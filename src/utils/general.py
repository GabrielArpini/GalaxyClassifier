from pathlib import Path
import sys 

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import os
import numpy as np
from astroNN.datasets import load_galaxy10 
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

import pickle

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


def get_data(marker='src',cleaned=True):
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
    cleaned_path = os.path.join(path, 'images_cleaned.npy')

    # First checks if there is cleaned images file
    if cleaned:
        if os.path.exists(cleaned_path) and os.path.exists(labels_path):
            print("Downloading cleaned images:")
            images = np.load(cleaned_path)
            labels = np.load(labels_path)
            return images,labels

    if os.path.exists(images_path) and os.path.exists(labels_path):
        print("No cleaned images found, please pre process them. Downloading normal images instead.")
        images = np.load(images_path)
        labels = np.load(labels_path)
    else:
        print("No images file found. Downloading from source.")
        images, labels = load_galaxy10()
        os.makedirs(path, exist_ok=True)
        np.save(images_path, images)
        np.save(labels_path, labels)
        
    
    return images, labels

def split_data_indices(images,labels, root_path, save=True):
    images_path = root_path / 'data' / 'images.npy'
    labels_path = root_path / 'data' / 'labels.npy'

    original_indices = np.arange(len(images))

    #split
    train_indices,_indices = train_test_split(original_indices, test_size=0.4, stratify=labels, random_state=42)
    valid_indices,test_indices = train_test_split(_indices,test_size=0.5, stratify=labels[_indices], random_state=42)
    
    assert len(set(train_indices).intersection(set(_indices))) == 0, "Train and _indices indices overlap!"
    assert len(set(valid_indices).intersection(set(test_indices))) == 0, "Test and validation indices overlap!"

    del original_indices
    del _indices

    # Save
    if save:
        train_indice_path = root_path / 'data' / 'train_indices.npy'
        valid_indice_path = root_path / 'data' / 'valid_indices.npy'
        test_indice_path = root_path / 'data' / 'test_indices.npy'
        if os.path.exists(train_indice_path) and os.path.exists(valid_indice_path) and os.path.exists(test_indice_path):
            print("paths already exists, breaking")
        else:
            print("saving...")
            np.save(train_indice_path, train_indices)
            np.save(valid_indice_path, valid_indices)
            np.save(test_indice_path, test_indices)

    return train_indices,valid_indices,test_indices

def get_splits(root_path,splits=["train","valid","test"]):
    """
    A function that gets the data indices, which was made with the objective of making
    the indices accessible in the entire project.
    
    Args:
        root_path(Path): The root path of the project
        splits(list): A set with the splits to get
    Returns:
        loaded_splits(dict): A dict with the loaded pickle files.
        None: If an error occur
    """
        
    # Create splits if doesnt exist

    images,labels = get_data()
    _,_,_ = split_data_indices(images,labels, root_path, save=True)

    # Main data path
    data_path = root_path / 'data'
    # Each split path
    split_files = {
        "train": data_path / 'train_indices.npy',
        "valid": data_path / 'valid_indices.npy',
        "test": data_path / 'test_indices.npy'
    }
    loaded_splits = {}
    # Handle splits if it is just one to load
    if len(splits) == 1:
        try:
            loaded_splits[splits[0]] = np.load(split_files[splits[0]])
            return loaded_splits
        except Exception as e:
            print(f"Error while loading single split: {e}")
            return None
    for split in splits:
        try:
            loaded_splits[split] = np.load(split_files[split])
        except Exception as e:
            print(f"An error occured while loading the splits: {e}")
            return None
    return loaded_splits
  
class LazyGalaxyDataset(Dataset):
    """
    Creates the Dataset with the .npy files with galaxy ndarray and label ndarray.
    It uses a lazy approach because my RAM (16 GB) can't load it all when splitting data and runing model.
    It uses the indices and mmap_mode to read the data without loading the entire data with the DataLoaders.
    """
    def __init__(self, indices, image_path, label_path, symmetry_path, transform=None):
        self.indices = indices
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        self.symmetries = np.load(symmetry_path, mmap_mode="r")
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        # FIX: Convert the memmap slice to a standard numpy array
        image = np.array(self.images[actual_idx])

        label = self.labels[actual_idx]
        symmetry = self.symmetries[actual_idx]

        label = torch.tensor(label, dtype=torch.int64)
        symmetry = torch.tensor(symmetry, dtype=torch.float32)

        if len(image.shape) == 3 and image.shape[0] <= 4:
            image = image.transpose(1, 2, 0)


        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
    return image, label, symmetry

   



if __name__ == '__main__':
    get_data()
