from pathlib import Path
import sys 

root_path = Path.cwd().parent 
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
from utils.general import get_data, LazyGalaxyDataset,get_project_root
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle 
from scipy.ndimage import binary_dilation
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, center_of_mass
from scipy.optimize import minimize
import cv2
from scipy.ndimage import binary_dilation, label, gaussian_filter
from skimage.morphology import remove_small_objects

np.random.seed(42)
torch.manual_seed(42)



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

        np.save(train_indice_path, train_indices)
        np.save(valid_indice_path, valid_indices)
        np.save(test_indice_path, test_indices)

    return train_indices,valid_indices,test_indices


def star_removal(image, percentile_threshold=97.0, min_star_size=2, max_star_size=100):
    """Remove stars while preserving galaxy centers"""
    
    # Handle different input shapes

    if len(image.shape) == 3:
        if image.shape[-1] in [1, 3]:  # (H, W, C) format
            image = image.transpose(2, 0, 1)  # Convert to (C, H, W)
        
        if image.shape[0] == 3:  # RGB
            gray = 0.299*image[0] + 0.587*image[1] + 0.114*image[2]
        else:
            gray = image[0]
    else:  # (H, W)
        gray = image

    if len(image.shape) == 3:  # (C, H, W)
        if image.shape[0] == 3:  # RGB
            gray = 0.299*image[0] + 0.587*image[1] + 0.114*image[2]
        else:
            gray = image[0]
    else:  # (H, W)
        gray = image
    
    # Find bright pixels
    threshold = np.percentile(gray, percentile_threshold)
    bright_mask = gray > threshold
    
    # Remove small noise first
    bright_mask = remove_small_objects(bright_mask, min_size=min_star_size)
    
    # Label connected components
    labeled_objects, num_objects = label(bright_mask)
    
    # Identify stars vs galaxy components
    star_mask = np.zeros_like(bright_mask)
    
    for obj_id in range(1, num_objects + 1):
        obj_mask = labeled_objects == obj_id
        
        # Calculate object properties
        obj_size = np.sum(obj_mask)
        
        # Get bounding box
        y_coords, x_coords = np.where(obj_mask)
        if len(y_coords) == 0:
            continue
            
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        # Calculate aspect ratio and compactness
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        aspect_ratio = max(width, height) / min(width, height)
        
        # Calculate circularity (4π*area/perimeter²)
        perimeter = np.sum(binary_dilation(obj_mask) ^ obj_mask)
        if perimeter > 0:
            circularity = 4 * np.pi * obj_size / (perimeter ** 2)
        else:
            circularity = 0
        
      
        center_y, center_x = gray.shape[0] // 2, gray.shape[1] // 2
        obj_center_y, obj_center_x = np.mean(y_coords), np.mean(x_coords)
        distance_from_center = np.sqrt((obj_center_y - center_y)**2 + 
                                     (obj_center_x - center_x)**2)
        
        # Classify as star if:
        is_star = (
            obj_size < max_star_size and  # Small
            aspect_ratio < 2.0 and       # Roughly circular
            circularity > 0.3 and        # Compact
            distance_from_center > min(gray.shape) * 0.1  # Not at center
        )
        
        if is_star:
            star_mask |= obj_mask
    
    # Dilate star mask slightly to cover halos
    star_mask = binary_dilation(star_mask, iterations=2)
    
    # Replace masked pixels
    cleaned_image = image.copy()
    if len(image.shape) == 3:
        for c in range(image.shape[0]):
            cleaned_image[c][star_mask] = np.median(image[c][~star_mask])
    else:
        cleaned_image[star_mask] = np.median(image[~star_mask])
    
    return cleaned_image, star_mask

def get_img_assymetry(img):
    """
    Applies the symmetry formula from paper "The Symmetry, Color, and Morphology of Galaxies"
    I assume that every galaxy is in the center, so there is no need to create a function
    to get center of galaxy, i know it can introduce more bias, but first i want to see results
    if it is bad i will apply it.
    """
    h, w = img.shape[-2:]
    angle = 180
    center_x,center_y = h // 2, w // 2
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    if len(img.shape) == 3:
        rotated = np.zeros_like(img)
        for c in range(img.shape[0]):
            rotated[c] = cv2.warpAffine(img[c], rotation_matrix, (w, h), 
                                      flags=cv2.INTER_LINEAR, 
                                      borderMode=cv2.BORDER_REFLECT)
    else:
        rotated = cv2.warpAffine(img, rotation_matrix, (w, h), 
                               flags=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_REFLECT)
    

    diff = img - rotated
    if len(img.shape) == 3:
        assymetries = []
        for c in range(img.shape[0]):
            diff_squared = diff[c]**2
            img_squared = img[c]**2
            numerator = np.sum(diff_squared / 2)
            denominator = np.sum(img_squared)

            if denominator > 0:
                assymetry_sqred = numerator / denominator
                assymetry = np.sqrt(assymetry_sqred)
            else:
                assymetry = 0
            assymetries.append(assymetry)
        return np.mean(assymetries)



def preprocess_images():
    images, _ = get_data(cleaned=False)
    print(f"Original images shape: {images.shape}")  
    print(f"Original images dtype: {images.dtype}")
    cleaned_images = []
    assymetries = []
    print("Cleaning stars and calculating symmetries")
    for i, img in enumerate(images):
        cleaned_img,_ = star_removal(img)
        assymetry = get_img_assymetry(cleaned_img)
        cleaned_images.append(cleaned_img)
        assymetries.append(assymetry)
        if i == 0:
            print(cleaned_img.shape)
            print(cleaned_img.dtype)
    del images
    # Save path
    print("Saving files...")
    root_path = get_project_root()
    default_path = root_path / "data"
    cleaned_path = default_path / "images_cleaned.npy"
    assymetries_path = default_path / "assymetries.npy"
    np.save(cleaned_path, np.array(cleaned_images))
    np.save(assymetries_path, np.array(assymetries))

if __name__ == '__main__':
    preprocess_images()
