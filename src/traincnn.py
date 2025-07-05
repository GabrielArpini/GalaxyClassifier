import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torchmetrics
from torchinfo import summary
from torchvision import transforms

from collections import Counter

from utils.focal_loss import FocalLoss
import utils.general as g
from utils.general import LazyGalaxyDataset
from cnn import NeuralNet

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astroNN.datasets import load_galaxy10

import optuna
# override Optuna's default logging to ERROR only, to use optuna with mlflow
optuna.logging.set_verbosity(optuna.logging.ERROR)

import mlflow
from mlflow.models.signature import infer_signature # to create custom infer_signature
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("galaxy-classifier-experiment")

torch.manual_seed(42)
np.random.seed(42)
print(f"Pytorch version:{torch.__version__}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")
def get_label_count(labels):
  counts = np.unique(labels, return_counts=True)
  return counts[1].tolist()

def conf_matrix_eval(model):
    model.eval()
    metric = torchmetrics.classification.MulticlassConfusionMatrix(num_classes=10).to(device)

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)  # Raw logits [batch_size, 10]

            # Convert logits to predicted class indices
            _, y_pred_classes = torch.max(y_pred_logits, 1)

            metric.update(y_pred_classes, y_batch)  

    conf_matrix = metric.compute()

    # Calculate precision, recall, and F1-score from the confusion matrix
    for i, class_name in enumerate(class_names):
        tp = conf_matrix[i, i].item()
        fp = conf_matrix[:, i].sum().item() - tp
        fn = conf_matrix[i, :].sum().item() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Metrics for {class_name}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1_score:.4f}")






def train_book(model, optimizer, criterion, train_loader, n_epochs, device,scheduler=None,batch_size=32):
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    train_accuracies = []
    train_losses = []    
    with open("model_summary.txt", "w") as f:
        f.write(str(summary(model, input_size=(batch_size, 3, 256, 256))))
    mlflow.log_artifact("model_summary.txt")
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_valid_loss = 0.0
        accuracy.reset()
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)

            # Ensure y_batch is integer indices
            if y_batch.dtype != torch.long:
                y_batch = y_batch.long()
        
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            _, predicted = y_pred.max(1)
            accuracy.update(predicted, y_batch)
            loss.backward()
            optimizer.step()

            total += y_batch.size(0)
            
        
            if batch_idx % 100 == 0:
                batch_loss = loss.item()
                batch_acc = accuracy.compute().item() 
                mlflow.log_metrics(
                    {"batch_loss": batch_loss, "batch_accuracy": batch_acc},
                    step=epoch * len(train_loader) + batch_idx,
                )
        epoch_accuracy = accuracy.compute().item()
        train_losses.append(total_loss)
        train_accuracies.append(epoch_accuracy)
        if scheduler:
            scheduler.step()

        model.eval()
        val_accuracy.reset()
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                valid_loss = criterion(y_pred, y_batch)
                total_valid_loss += valid_loss.item()

 
                _, predicted_class = torch.max(y_pred, 1)  
        
                val_accuracy.update(predicted_class, y_batch)  

        val_acc_accuracy = val_accuracy.compute()
        mean_valid_loss = total_valid_loss / len(valid_loader)
        accuracy.reset()
        
        
     

        mean_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", f"{mean_loss:.6f}",step=epoch)
        mlflow.log_metric("train_accuracy",f"{epoch_accuracy:.6f}",step=epoch)
        mlflow.log_metric("validation_accuracy",f"{val_acc_accuracy:.6f}",step=epoch)
        mlflow.log_metric("validation_loss", f"{mean_valid_loss:.6f}", step=epoch)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {mean_loss:.6f}, Train Accuracy: {epoch_accuracy:.6f}, Validation Loss: {mean_valid_loss:.6f} Validation Accuracy: {val_acc_accuracy:.6f}")
    return model, val_acc_accuracy
def champion_callback(study, frozen_trial):
    """
    Implementation from official mlflow docs
    """
    winner = study.user_attrs.get("winner", None)
    
    if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
          "epochs": 30,
          "learning_rate": trial.suggest_float("lr", 1e-3, 1e-2, log=True),
          "batch_size": 32,
          "loss function": "Class Balanced Focal Loss",
          "fl_beta": trial.suggest_categorical("fl_beta",[0.9,0.99,0.999,0.9999]),
          "fl_gamma": trial.suggest_float("fl_gamma",1,10),
          "optimizer": "ADAM",
        }
        mlflow.log_params(params)

        model = NeuralNet().to(device)
        loss = FocalLoss(
            beta= params["fl_beta"],
            gamma=params["fl_gamma"],
            samples_per_class=num_samples_per_class,
            reduce=True,
            device=device
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
        # Train model
        print("training model...")
        model, validation_accuracy =train_book(model, optimizer, loss, train_loader, n_epochs=params["epochs"],device=device,scheduler=scheduler)
        
    
        
        model.eval()
        with torch.no_grad():
            # Ensure img_example is a single example, not a batch
            single_img_example = img_example[0:1].to(device)  # Shape: [1, 3, 256, 256]
            output = model(single_img_example)
        
        # Convert to NumPy arrays for MLflow
        input_example_np = single_img_example.cpu().detach().numpy()  # Shape: [1, 3, 256, 256]
        output_np = output.cpu().detach().numpy()  # Model output as NumPy
        
        # Verify input_example_np is a NumPy array
        if not isinstance(input_example_np, np.ndarray):
            raise ValueError(f"input_example_np is not a NumPy array, got {type(input_example_np)}")
        
        # Create signature
        signature = infer_signature(input_example_np, output_np)

        conf_matrix_eval(model)

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="cbfl_model",
            input_example=input_example_np,
            signature=signature,
            code_paths=code_paths 
        )
        return -validation_accuracy

class_names = [
    "0 - Disturbed Galaxies",
    "1 - Merging Galaxies",
    "2 - Round Smooth Galaxies",
    "3 - In-between Round Smooth Galaxies",
    "4 - Cigar Shaped Smooth Galaxies",
    "5 - Barred Spiral Galaxies",
    "6 - Unbarred Tight Spiral Galaxies",
    "7 - Unbarred Loose Spiral Galaxies",
    "8 - Edge-on Galaxies without Bulge",
    "9 - Edge-on Galaxies with Bulge"
]
root_path = g.get_project_root()

code_paths = [
            str(root_path / "src" / "utils" / "focal_loss.py"),
            str(root_path / "src" / "cnn.py")
        ]

images,labels = g.get_data()
images_path =  root_path / 'data' / 'images.npy'
labels_path =  root_path / 'data' / 'labels.npy'

original_indices = np.arange(len(images))

train_indices, temp_indices = train_test_split(
    original_indices, test_size=0.4, stratify=labels, random_state=42
)

val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=labels[temp_indices], random_state=42
)


# Create transformations for data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
])
train_transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.ToTensor(),  
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
train_dataset = LazyGalaxyDataset(train_indices,images_path,labels_path, train_transform)
test_dataset = LazyGalaxyDataset(test_indices,images_path,labels_path)
valid_dataset = LazyGalaxyDataset(val_indices, images_path, labels_path, val_transform)




train_labels = labels[train_indices]
num_samples_per_class = get_label_count(train_labels)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

img_example = next(iter(train_loader))[0]

# Try CE loss instead of FL
weights = torch.tensor([1.0 / s for s in num_samples_per_class], device=device)
weights = weights / weights.max()  # Normalize to cap weights
#loss = nn.CrossEntropyLoss(weight=weights)


# Main training loop with optuna
with mlflow.start_run(nested=True) as run:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, callbacks=[champion_callback])    
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation accuracy: {-best_trial.value:.6f}")
    print(f"Best parameters: {best_trial.params}") 

torch.cuda.empty_cache()






