

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.focal_loss import FocalLoss
import utils.general as g
from utils.general import *
from utils.early_stop import EarlyStopping
from cnn import NeuralNet

import os
from pathlib import Path
import numpy as np

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

import mlflow
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("galaxy-classifier-experiment")

# Set up seeds for deterministic results 
torch.manual_seed(42)
np.random.seed(42)
print(f"Pytorch version:{torch.__version__}")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using device: {device}")

# Helper function 
def get_label_count(labels):
    counts = np.unique(labels, return_counts=True)
    return counts[1].tolist()

def train_book(model, optimizer, criterion, train_loader, n_epochs, device,optuna_trial, scheduler=None, batch_size=32):

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    train_accuracies = []

    train_losses = []

    early_stopping = EarlyStopping(patience=15, min_delta=0.0005)



   

    

    image_input = torch.randn(batch_size, 3, 256, 256).to(device)

    symmetry_input = torch.randn(batch_size).to(device)

    # Commented out because summary doesnt work well with geometric tensors from escnn

    #summary_str = summary(model, input_data=image_input, verbose=1)

    model_info = f"""

        Model Architecture:

        - Type: Multi-modal Equivariant CNN (ESCNN + MLP)

        - Group: C4 (4-fold rotation)

        - Inputs:

            - Image: ({batch_size} x 3 x 256 x 256)

            - Symmetry: ({batch_size} x 1)

        - Classes: 10

        - Optimizer: {type(optimizer).__name__}

        - Scheduler: {type(scheduler).__name__ if scheduler else 'None'}

        """ 

      



    with open("model_summary.txt", "w") as f:

        f.write(model_info)

    mlflow.log_artifact("model_summary.txt")

    

    

    for epoch in range(n_epochs):

        model.train()

        total_loss = 0.0

        total_valid_loss = 0.0

        accuracy.reset()

        total = 0

            

        for batch_idx, (X_batch, y_batch, symmetry_batch) in enumerate(train_loader):

            X_batch, y_batch, symmetry_batch = X_batch.to(device), y_batch.to(device), symmetry_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch, symmetry_batch)



            # Ensure y_batch is integer indices

            if y_batch.dtype != torch.long:

                y_batch = y_batch.long()

        

            loss = criterion(y_pred, y_batch)

            total_loss += loss.item()



            _, predicted = y_pred.max(1)

            accuracy.update(predicted, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) #Gradient clipping 

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

        



        model.eval()

        val_accuracy.reset()

        with torch.no_grad():

            for X_batch, y_batch,symmetry_batch in valid_loader:

                X_batch, y_batch, symmetry_batch = X_batch.to(device), y_batch.to(device), symmetry_batch.to(device)

                y_pred = model(X_batch, symmetry_batch)

                valid_loss = criterion(y_pred, y_batch)

                total_valid_loss += valid_loss.item()



 

                _, predicted_class = torch.max(y_pred, 1)  

        

                val_accuracy.update(predicted_class, y_batch)  

        mean_loss = total_loss / len(train_loader)

        val_acc_accuracy = val_accuracy.compute()

        mean_valid_loss = total_valid_loss / len(valid_loader)

        accuracy.reset()



        total_grad_norm = 0

        for p in model.parameters():

            if p.grad is not None:

                total_grad_norm += p.grad.data.norm(2).item() ** 2

        total_grad_norm = total_grad_norm ** 0.5

        print(f"Gradient norm: {total_grad_norm:.4f}")

        

        #early_stopping(mean_valid_loss)

        #if early_stopping.early_stop:

            #print("Early stopping at epoch:", epoch)

            #break  



        if scheduler:

            # Add param 'val_acc_accuracy' if scheduler is Plateau

            scheduler.step(val_acc_accuracy)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}, LR: {current_lr:.2e}")

        mlflow.log_metric("gradient_norm",total_grad_norm, step=epoch)

        mlflow.log_metric("learning_rate", current_lr, step=epoch)

        mlflow.log_metric("train_loss", mean_loss, step=epoch)

        mlflow.log_metric("train_accuracy", epoch_accuracy, step=epoch)

        mlflow.log_metric("validation_accuracy", val_acc_accuracy.item(), step=epoch)

        mlflow.log_metric("validation_loss", mean_valid_loss, step=epoch)



        # Optuna reporting

        optuna_trial.report(val_acc_accuracy.item(), epoch)

        if optuna_trial.should_prune():

            raise optuna.exceptions.TrialPruned()



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

    with mlflow.start_run(nested=True) as nested_run:

        

        # Optimizer and scheduler parameters

        learning_rate = 0.0040175444467859785 #trial.suggest_float("lr", 1e-5, 1e-2, log=True)

        weight_decay = 0.008883602189886384  #trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

        

        # Coswarm parameters

        #scheduler_T0 = trial.suggest_int("scheduler_T0", 5, 20)

        #scheduler_Tmult = trial.suggest_int("scheduler_Tmult", 1, 2)

        #scheduler_eta_min = trial.suggest_float("scheduler_eta_min", 1e-8, 1e-5, log=True)
    
        # ReduceLRonPlateau params 
        factor = 0.3
        
        patience= 7
        


        # Focal Loss parameters

        fl_gamma = 1.0067947459080258  #trial.suggest_float("fl_gamma", 1.0, 5.0)

        fl_beta = 0.9308672937503113 #trial.suggest_float("fl_beta", 0.9, 0.9999)



        params = {

            "epochs": 250,

            "learning_rate": learning_rate,

            "weight_decay": weight_decay,

            "batch_size": 16,

            "loss_function": "Class Balanced Focal Loss",

            "fl_gamma": fl_gamma,

            "fl_beta": fl_beta,

            "optimizer": "AdamW",

            "scheduler": "ReduceLRonPlateau",
            
            "factor": factor,

            "patience": patience, 

            #"scheduler_T0": scheduler_T0,

            #"scheduler_Tmult": scheduler_Tmult,

            #"scheduler_eta_min": scheduler_eta_min,

        }





        

        

        mlflow.log_params(params)

        model = NeuralNet().to(device)

        

        # Rewrote focal loss, optimizer, scheduler for more legibility.

        loss_fn = FocalLoss(

            beta=params["fl_beta"],

            gamma=params["fl_gamma"],

            samples_per_class=num_samples_per_class,

            device=device

        ).to(device)

        

        optimizer = torch.optim.AdamW(

            model.parameters(), 

            lr=params["learning_rate"], 

            weight_decay=params["weight_decay"]

        )

        
        
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(

            #optimizer,

            #T_0=params["scheduler_T0"],

            #T_mult=params["scheduler_Tmult"],

            #eta_min=params["scheduler_eta_min"]

        #)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=factor,
            patience=patience
        )



       



        # Try CE loss instead of FL to check if training issues are from FL.

        

        #total_samples = sum(num_samples_per_class)

        #weights = torch.tensor([

        #    total_samples / (len(num_samples_per_class) * s) 

        #    for s in num_samples_per_class

        #], device=device)

        #weights = weights / weights.sum() * len(weights)

        #loss = nn.CrossEntropyLoss(weight=weights)

        

       



        # Train model

        # (model, optimizer, criterion, train_loader, n_epochs, device,trial, scheduler=None, batch_size=16):

        print("training model...")

        model, validation_accuracy = train_book(

            model, optimizer, loss_fn, train_loader,

            n_epochs=params["epochs"],

            device=device,

            optuna_trial=trial, # Pass trial for the optuna reporting

            scheduler=scheduler, 

            batch_size=params["batch_size"]

        )

    

        

        model.eval()

        with torch.no_grad():

            # Ensure img_example is a single example, not a batch

            img_example,_, symmetries = next(iter(train_loader))

            single_img_example = img_example[0:1].to(device)  # Shape: [1, 3, 256, 256]

            single_symmetry_example = symmetries[0:1].to(device)

            output = model(single_img_example, single_symmetry_example)

        

       

        

        input_example = {

            "image_input": single_img_example.cpu().detach().numpy(),

            "symmetry_input": single_symmetry_example.cpu().detach().numpy()

        }

        output_np = output.cpu().detach().numpy()



    

        signature = infer_signature(input_example, output_np)



    

        model_info = mlflow.pytorch.log_model(

            pytorch_model=model,

            artifact_path="cbfl_model",

            signature=signature,

            #input_example=input_example, # Generates a warning about the input type not supported for pytorch flavor, need fix if used. 

            code_paths=code_paths

        )

        

        

        trial.set_user_attr("model_uri", model_info.model_uri)

        trial.set_user_attr("run_id", nested_run.info.run_id)

        return -validation_accuracy.item()



def register_best_model(study, model_name="galaxy-classifier"):

    """

    Register the best model from the study in the MLflow model registry.



    Parameters:

    study: Optuna optimization result.

    model_name: The name to register the model in the MLflow model registry.



    Returns:

    best_model: MLflow ModelVersion object with best model information, or None if registration fails or is not needed.

    """

    # Get the best trial from Optuna

    best_trial = study.best_trial

    best_model_uri = best_trial.user_attrs.get("model_uri")

    if not best_model_uri:

        print("No model URI found in best trial")

        return None



    try:

        client = MlflowClient()



        # Ensure the model is registered in the registry

        try:

            client.get_registered_model(model_name)

        except mlflow.exceptions.MlflowException:

            client.create_registered_model(

                name=model_name,

                description="Galaxy classifier model trained with Optuna for hyperparameter optimization."

            )



        # Prepare description for the new model version

        description = (

            f"Galaxy classifier - Best trial {best_trial.number}. "

            f"Validation accuracy: {-best_trial.value:.6f}. "

            f"Parameters: {best_trial.params}"

        )



        # Get current accuracy (negated because Optuna minimizes objective)

        current_accuracy = -best_trial.value



        # Search for the latest model version with the "production" alias

        versions = client.search_model_versions(f"name='{model_name}'")

        production_version = None

        for version in versions:

            if "production" in client.get_model_version(model_name, version.version).aliases:

                production_version = version

                break



        # Compare with the existing production model's accuracy

        best_accuracy = None

        if production_version:

            run_id = production_version.run_id

            run = client.get_run(run_id)

            best_accuracy = run.data.metrics.get("validation_accuracy")



        # Register the new model if no production version exists or if it has better accuracy

        if best_accuracy is None or current_accuracy > best_accuracy:

            best_model = client.create_model_version(

                name=model_name,

                source=best_model_uri,

                run_id=best_trial.user_attrs.get("run_id"),

                description=description

            )



            # Set the "production" alias for the new model version

            client.set_registered_model_alias(

                name=model_name,

                alias="production",

                version=best_model.version

            )



            print(f"Best model registered: {model_name} v{best_model.version} with alias 'production'")

            return best_model

        else:

            print(f"Current model (accuracy: {current_accuracy:.6f}) not better than existing production model (accuracy: {best_accuracy:.6f})")

            return None



    except Exception as e:

        print(f"Error registering best model: {e}")

        return None









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

dict_indices = g.get_splits(root_path,splits=["train","valid"])

train_indices = dict_indices["train"]

val_indices = dict_indices["valid"]



# data path

images_path = root_path / "data" / "images_cleaned.npy"

labels_path = root_path / "data" / "labels.npy"

symmetry_path = root_path / "data" / "assymetries.npy"



# For mlflow

code_paths = [

            str(root_path / "src" / "utils" / "focal_loss.py"),

            str(root_path / "src" / "cnn.py")

        ]



# Create transformations for data augmentation



train_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),

    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    #transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=5),

    #transforms.RandomHorizontalFlip(p=0.5),

    #transforms.RandomRotation(degrees=180, interpolation=transforms.InterpolationMode.BILINEAR, expand=False), 

    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  

])



val_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])



train_dataset = LazyGalaxyDataset(train_indices,images_path,labels_path,symmetry_path, transform=train_transform)

valid_dataset = LazyGalaxyDataset(val_indices, images_path, labels_path,symmetry_path, transform=val_transform)







_,labels = g.get_data()

train_labels = labels[train_indices]

num_samples_per_class = get_label_count(train_labels)



batch_size = 16



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6,pin_memory=True, drop_last=True) # drop_last solves error from batch norm, since it drop incomplete batch

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)



img_example = next(iter(train_loader))[0]



# Try CE loss instead of FL

weights = torch.tensor([1.0 / s for s in num_samples_per_class], device=device)

weights = weights / weights.max()  # Normalize to cap weights

#loss = nn.CrossEntropyLoss(weight=weights)





# Main training loop with optuna



with mlflow.start_run(run_name="GalaxyClassifier_Hyperparameter_Search") as parent_run:

    print("Starting Optuna hyperparameter search with HyperbandPruner...")



    study = optuna.create_study(

        direction="minimize", 

        pruner=optuna.pruners.HyperbandPruner(

            min_resource=5,          

            max_resource=100,       

            reduction_factor=3       

        )

    )



    try:

        study.optimize(

            objective,

            n_trials=1, 

            callbacks=[champion_callback] 

        )

    except Exception as e:

        print(f"An error occurred during optimization: {e}")





    print("\n Hyperparameter search complete.")





    if len(study.trials) > 0 and study.best_trial is not None:

        best_trial = study.best_trial

        print("\n--- Best Trial Results ---")

        print(f"  Trial Number: {best_trial.number}")



        print(f"  Validation Accuracy: {-best_trial.value:.6f}")

        print("  Best Parameters Found:")

        for key, value in best_trial.params.items():

            print(f"    {key}: {value:.6f}" if isinstance(value, float) else f"    {key}: {value}")



       

        best_model_version = register_best_model(study, model_name="galaxy-classifier-ecnn")

    else:

        print("No successful trials were completed.")



# Clean CUDA.

torch.cuda.empty_cache()




