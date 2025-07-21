# Galaxy Classifier CNN with class balanced focal loss
The following project is an attempt to implement a Convolutional Neural Network using a custom made loss function from the paper: "Class-Balanced Loss Based on Effective Number of Samples" from Yin Cui et al.

## Introduction

The main objective of this project is to achieve a solid grasp in the Pytorch framework with a hands-on approach. As I am a huge enthusiast of astrophysics, i've decided to do my project in that field. 
In our universe there are several astronomical objects, like: planets, suns, asteroids, black holes, galaxies, etc. Because of their different formation process, they can achieve different morphologies that can be observed from our planet, but because everything in our universe follows the rules of physics, it is possible to find some patterns in said objects, the easiest one to grasp (at least for me) is the morphology of galaxies, they can have multiple spirals, not have spirals at all, etc. This project main goal is to create a CNN capable of classifying an image of a galaxy with certain accuracy above an arbitrary threshold defined by my, which is 80%, since SOTA models can achieve above 90% accuracy using different methodologies.

## Device specs
Bellow is the computer specs of my PC, used to do the project, a Google Colab will do just fine, but it blocks the GPU after a few hours so i decided to use my PC:

GPU: RTX 2060 6GB
CPU: Ryzen 5 5600
RAM: 2x8GB DDR4 
OS: Ubuntu 24.04.2 LTS


## Dataset
The dataset used is from the astronn paper from Henry W. Leung and Jo Bovy, called Galaxy10 DECaLS Dataset, which combines the GalaxyZoo Data Release 2 with GZ DR2 with DECaLS images instead of SDSS images, which have a better resolution with about 18.000 images selected with 10 classifications:

Galaxy10 dataset (17736 images)
├── Class 0 (1081 images): Disturbed Galaxies
├── Class 1 (1853 images): Merging Galaxies
├── Class 2 (2645 images): Round Smooth Galaxies
├── Class 3 (2027 images): In-between Round Smooth Galaxies
├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies
├── Class 5 (2043 images): Barred Spiral Galaxies
├── Class 6 (1829 images): Unbarred Tight Spiral Galaxies
├── Class 7 (2628 images): Unbarred Loose Spiral Galaxies
├── Class 8 (1423 images): Edge-on Galaxies without Bulge
└── Class 9 (1873 images): Edge-on Galaxies with Bulge
Taken from official website: https://astronn.readthedocs.io/en/latest/galaxy10.html#introduction

Because the dataset is huge to load at once, and then split it into training dataset and validation dataset for my RAM or Google Colab, i implemented a Lazy Load approach, which stores in ram the indices of the galaxy images/labels, and loads them as needed, which saved a lot of space, sadly my Ubuntu is in a HDD, which made things slow for me, but at least didn't overloaded the memory.

## Project structure

GalaxyClassifier
├── data # Folder with galaxies and labels.
│   ├── images.npy # Downloaded with the train script
│   └── labels.npy # Downloaded with the train script
├── LICENSE
├── mlflow.db # sqlite3 database to store model runs and info.
├── model_summary.txt # Model structure summary
├── notebooks 
│   ├── cam_visualization.png # Cam image from analysis notebook
│   └── model_analysis.ipynb # Notebook with post training analysis
├── README.md
├── requirements.txt 
└── src
    ├── cnn.py # The CNN structure
    ├── traincnn.py # The training script
    └── utils # Utilities directory 
        ├── early_stop.py # Class for the early stop feature
        ├── focal_loss.py # Class balanced focal loss implementation
        ├── general.py # General utilities.
        └── \_\_init\_\_.py

5 directories, 16 files


# CNN (Convolutional Neural Network)
TODO INSERT IMAGE OR MAINTAIN REPRESENTATION BELLOW

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
NeuralNet                                [32, 10]                  --
├─Conv2d: 1-1                            [32, 32, 256, 256]        896
├─MaxPool2d: 1-2                         [32, 32, 128, 128]        --
├─Conv2d: 1-3                            [32, 64, 128, 128]        18,496
├─MaxPool2d: 1-4                         [32, 64, 64, 64]          --
├─Conv2d: 1-5                            [32, 128, 64, 64]         73,856
├─MaxPool2d: 1-6                         [32, 128, 32, 32]         --
├─Conv2d: 1-7                            [32, 256, 32, 32]         295,168
├─MaxPool2d: 1-8                         [32, 256, 16, 16]         --
├─Conv2d: 1-9                            [32, 512, 16, 16]         1,180,160
├─MaxPool2d: 1-10                        [32, 512, 8, 8]           --
├─Conv2d: 1-11                           [32, 1024, 8, 8]          4,719,616
├─MaxPool2d: 1-12                        [32, 1024, 4, 4]          --
├─Linear: 1-13                           [32, 512]                 8,389,120
├─Dropout: 1-14                          [32, 512]                 --
├─Linear: 1-15                           [32, 256]                 131,328
├─Dropout: 1-16                          [32, 256]                 --
├─Linear: 1-17                           [32, 10]                  2,570
==========================================================================================
Total params: 14,811,210
Trainable params: 14,811,210
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 50.54
==========================================================================================
Input size (MB): 25.17
Forward/backward pass size (MB): 1057.16
Params size (MB): 59.24
Estimated Total Size (MB): 1141.57
==========================================================================================

The model also uses kaiming initialization from Kaiming He et al., which was implemented to help prevent exploding gradient problem that i faced during implementation.

# Training process optimizations

With the goal to check model performance and compare hyperparameters i've decided to implement a MLflow pipeline, to save the experiments in a nested loop used by Optuna (Hyperparameter optimizer that takes a lot of time to find meaningful hyperparameters), the MLflow pipeline also stores the best model in the model registry, making it possible to load it easily in other files, such as the `model_analysis.ipynb`. The best model is achieved after `n` iterations from Optuna, returning the model with best validation accuracy, i've ran Optuna multiple times, with some runs that took more than 30 hours, sadly i made a mistake in the model optimization which made me lose the value of one of them (Focal loss was not able to handle the values of the pt calculation causing NaN after around 80 epochs in one of them, the other was simply a bad model that i used). The training process uses gradient clipping to help with exploding gradients as well.

The galaxy dataset contains some classes with a huge imbalance if compared with the mean, to solve that i've implemented the Class Balanced Focal Loss, which computes the effective number of samples and uses it to attribute a focus to the under presented classes with hyperparameters tuned by Optuna.
Data Augmentation is also used, the following code shows all the augmentation used, to help with the imbalanced classes and the underfitting problem:

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop((170,240)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5),
    v2.GaussianBlur(kernel_size=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0,270)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

RADAM is used as an optimizer, as it doesn't need to warm up the weights or tune hyperparameters, it is used with a OneCycleLR scheduler, allowing the model to escape local minima (Huge problem that i had, and probably still have in some magnitude).



temp_references: https://arxiv.org/pdf/1901.05555
https://arxiv.org/pdf/1908.03265
https://arxiv.org/pdf/1502.01852
https://arxiv.org/pdf/1912.01703
https://arxiv.org/pdf/1511.08458
https://arxiv.org/pdf/1207.0580
https://arxiv.org/pdf/1809.01691

astronn official github: https://github.com/henrysky/astroNN
astronn offical paper: https://arxiv.org/pdf/1808.04428
