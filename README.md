# Galaxy Classifier CNN with class balanced focal loss
The following project is an attempt to implement a Convolutional Neural Network using a custom made loss function from the paper: "Class-Balanced Loss Based on Effective Number of Samples" from Yin Cui et al.

## Introduction

The main objective of this project is to achieve a solid grasp in the Pytorch framework with a hands-on approach. As I am a huge enthusiast of astrophysics, i've decided to do my project in that field. 
In our universe there are several astronomical objects, like: planets, suns, asteroids, black holes, galaxies, etc. Because of their different formation process, they can achieve different morphologies that can be observed from our planet, but because everything in our universe follows the rules of physics, it is possible to find some patterns in said objects, the easiest one to grasp (at least for me) is the morphology of galaxies, they can have multiple spirals, not have spirals at all, etc. This project main goal is to create a CNN capable of classifying an image of a galaxy with certain accuracy above an arbitrary threshold defined by my, which is 80%, since SOTA models can achieve above 90% accuracy using different methodologies.

## Device specs
Bellow is the computer specs of my PC, used to do the project, a Google Colab will do just fine, but it blocks the GPU after a few hours so i decided to use my PC:

GPU: RTX 2060 6GB <br>
CPU: Ryzen 5 5600 <br>
RAM: 2x8GB DDR4 <br>
OS: Ubuntu 24.04.2 LTS <br>
Pyton 3.11 


## Dataset
The dataset used is from the astronn paper from Henry W. Leung and Jo Bovy, called Galaxy10 DECaLS Dataset, which combines the GalaxyZoo Data Release 2 with GZ DR2 with DECaLS images instead of SDSS images, which have a better resolution with about 18.000 images selected with 10 classifications:

Galaxy10 dataset (17736 images) <br>
├── Class 0 (1081 images): Disturbed Galaxies <br>
├── Class 1 (1853 images): Merging Galaxies <br>
├── Class 2 (2645 images): Round Smooth Galaxies <br>
├── Class 3 (2027 images): In-between Round Smooth Galaxies <br>
├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies <br>
├── Class 5 (2043 images): Barred Spiral Galaxies <br>
├── Class 6 (1829 images): Unbarred Tight Spiral Galaxies <br>
├── Class 7 (2628 images): Unbarred Loose Spiral Galaxies <br>
├── Class 8 (1423 images): Edge-on Galaxies without Bulge <br>
└── Class 9 (1873 images): Edge-on Galaxies with Bulge <br>
Taken from official website: https://astronn.readthedocs.io/en/latest/galaxy10.html#introduction <br>

Because the dataset is huge to load at once, and then split it into training dataset and validation dataset for my RAM or Google Colab, i implemented a Lazy Load approach, which stores in ram the indices of the galaxy images/labels, and loads them as needed, which saved a lot of space, sadly my Ubuntu is in a HDD, which made things slow for me, but at least didn't overloaded the memory.

## Project structure

GalaxyClassifier <br>
├── data # Folder with galaxies and labels. <br>
│   ├── images.npy # Downloaded with the train script <br>
│   └── labels.npy # Downloaded with the train script <br>
├── LICENSE <br>
├── mlflow.db # sqlite3 database to store model runs and info. <br>
├── model_summary.txt # Model structure summary <br>
├── notebooks  <br>
│   ├── cam_visualization.png # Cam image from analysis notebook <br>
│   └── model_analysis.ipynb # Notebook with post training analysis <br>
├── README.md <br>
├── requirements.txt <br> 
└── src <br>
    ├── cnn.py # The CNN structure <br>
    ├── traincnn.py # The training script <br>
    └── utils # Utilities directory  <br>
        ├── early_stop.py # Class for the early stop feature <br>
        ├── focal_loss.py # Class balanced focal loss implementation <br>
        ├── general.py # General utilities. <br>
        └── \_\_init\_\_.py <br>

5 directories, 16 files <br>


# CNN (Convolutional Neural Network)
TODO INSERT IMAGE OR MAINTAIN REPRESENTATION BELLOW

| Layer (type:depth-idx) | Output Shape            | Param #   |
|---------------------|-------------------------|-----------|
| NeuralNet           | [32, 10]                | --        |
| Conv2d: 1-1         | [32, 32, 256, 256]      | 896       |
| MaxPool2d: 1-2      | [32, 32, 128, 128]      | --        |
| Conv2d: 1-3         | [32, 64, 128, 128]      | 18,496    |
| MaxPool2d: 1-4      | [32, 64, 64, 64]        | --        |
| Conv2d: 1-5         | [32, 128, 64, 64]       | 73,856    |
| MaxPool2d: 1-6      | [32, 128, 32, 32]       | --        |
| Conv2d: 1-7         | [32, 256, 32, 32]       | 295,168   |
| MaxPool2d: 1-8      | [32, 256, 16, 16]       | --        |
| Conv2d: 1-9         | [32, 512, 16, 16]       | 1,180,160 |
| MaxPool2d: 1-10     | [32, 512, 8, 8]         | --        |
| Conv2d: 1-11        | [32, 1024, 8, 8]        | 4,719,616 |
| MaxPool2d: 1-12     | [32, 1024, 4, 4]        | --        |
| Linear: 1-13        | [32, 512]               | 8,389,120 |
| Dropout: 1-14       | [32, 512]               | --        |
| Linear: 1-15        | [32, 256]               | 131,328   |
| Dropout: 1-16       | [32, 256]               | --        |
| Linear: 1-17        | [32, 10]                | 2,570     |

## Summary Statistics
- **Total params**: 14,811,210
- **Trainable params**: 14,811,210
- **Non-trainable params**: 0
- **Total mult-adds (Units.GIGABYTES)**: 50.54
- **Input size (MB)**: 25.17
- **Forward/backward pass size (MB)**: 1057.16
- **Params size (MB)**: 59.24
- **Estimated Total Size (MB)**: 1141.57

The model also uses kaiming initialization from Kaiming He et al., which was implemented to help prevent exploding gradient problem that i faced during implementation.

# Training process optimizations

With the goal to check model performance and compare hyperparameters i've decided to implement a MLflow pipeline, to save the experiments in a nested loop used by Optuna (Hyperparameter optimizer that takes a lot of time to find meaningful hyperparameters), the MLflow pipeline also stores the best model in the model registry, making it possible to load it easily in other files, such as the `model_analysis.ipynb`. The best model is achieved after `n` iterations from Optuna, returning the model with best validation accuracy, i've ran Optuna multiple times, with some runs that took more than 30 hours, sadly i made a mistake in the model optimization which made me lose the value of one of them (Focal loss was not able to handle the values of the pt calculation causing NaN after around 80 epochs in one of them, the other was simply a bad model that i used). The training process uses gradient clipping to help with exploding gradients as well.

The galaxy dataset contains some classes with a huge imbalance if compared with the mean, to solve that i've implemented the Class Balanced Focal Loss, which computes the effective number of samples and uses it to attribute a focus to the under presented classes with hyperparameters tuned by Optuna.
Data Augmentation is also used, the following code shows all the augmentation used, to help with the imbalanced classes and the underfitting problem:

```Python
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
```
RADAM is used as an optimizer, as it doesn't need to warm up the weights or tune hyperparameters, it is used with a OneCycleLR scheduler, allowing the model to escape local minima (Huge problem that i had, and probably still have in some magnitude).

# How to run
First, clone the repository and cd into it:
```Bash
git clone https://github.com/GabrielArpini/GalaxyClassifier.git
cd GalaxyClassifier
```
Then you need to start mlflow with:
```Bash
$ mlflow ui --backend-store-uri sqlite:///mlflow.db
```
This will open the port 5000 for the UI of mlflow, where you will be able to see the experiment runs and model version, and also enable the python scripts to comunicate with the API.
Now, you can run the `traincnn.py` script to start the training process, you can tweak the training variables inside the `traincnn.py` file.
Before running the script make sure that you have gcc-fortran installend, since escnn uses it.
```Bash
$ python3 src/traincnn.py
```
The folder `notebooks` contains some analysis (for now a CAM image and confusion matrix) of the best model saved in the mlflow model registry, to run that you need to just start the jupyter notebook and run the cells:
```Bash
$ jupyter notebook
```

# Citations
```bibtex
@article{DBLP:journals/corr/abs-1901-05555,
  author       = {Yin Cui and
                  Menglin Jia and
                  Tsung{-}Yi Lin and
                  Yang Song and
                  Serge J. Belongie},
  title        = {Class-Balanced Loss Based on Effective Number of Samples},
  journal      = {CoRR},
  volume       = {abs/1901.05555},
  year         = {2019},
  url          = {http://arxiv.org/abs/1901.05555},
  eprinttype    = {arXiv},
  eprint       = {1901.05555},
  timestamp    = {Tue, 08 Sep 2020 16:29:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1901-05555.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/abs-1908-03265,
  author       = {Liyuan Liu and
                  Haoming Jiang and
                  Pengcheng He and
                  Weizhu Chen and
                  Xiaodong Liu and
                  Jianfeng Gao and
                  Jiawei Han},
  title        = {On the Variance of the Adaptive Learning Rate and Beyond},
  journal      = {CoRR},
  volume       = {abs/1908.03265},
  year         = {2019},
  url          = {http://arxiv.org/abs/1908.03265},
  eprinttype    = {arXiv},
  eprint       = {1908.03265},
  timestamp    = {Thu, 11 Apr 2024 13:33:57 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1908-03265.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/HeZR015,
  author       = {Kaiming He and
                  Xiangyu Zhang and
                  Shaoqing Ren and
                  Jian Sun},
  title        = {Delving Deep into Rectifiers: Surpassing Human-Level Performance on
                  ImageNet Classification},
  journal      = {CoRR},
  volume       = {abs/1502.01852},
  year         = {2015},
  url          = {http://arxiv.org/abs/1502.01852},
  eprinttype    = {arXiv},
  eprint       = {1502.01852},
  timestamp    = {Wed, 25 Jan 2023 11:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/HeZR015.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@article{DBLP:journals/corr/abs-1912-01703,
  author       = {Adam Paszke and
                  Sam Gross and
                  Francisco Massa and
                  Adam Lerer and
                  James Bradbury and
                  Gregory Chanan and
                  Trevor Killeen and
                  Zeming Lin and
                  Natalia Gimelshein and
                  Luca Antiga and
                  Alban Desmaison and
                  Andreas K{\"{o}}pf and
                  Edward Z. Yang and
                  Zach DeVito and
                  Martin Raison and
                  Alykhan Tejani and
                  Sasank Chilamkurthy and
                  Benoit Steiner and
                  Lu Fang and
                  Junjie Bai and
                  Soumith Chintala},
  title        = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  journal      = {CoRR},
  volume       = {abs/1912.01703},
  year         = {2019},
  url          = {http://arxiv.org/abs/1912.01703},
  eprinttype    = {arXiv},
  eprint       = {1912.01703},
  timestamp    = {Tue, 02 Nov 2021 15:18:32 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1912-01703.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{DBLP:journals/corr/OSheaN15,
  author       = {Keiron O'Shea and
                  Ryan Nash},
  title        = {An Introduction to Convolutional Neural Networks},
  journal      = {CoRR},
  volume       = {abs/1511.08458},
  year         = {2015},
  url          = {http://arxiv.org/abs/1511.08458},
  eprinttype    = {arXiv},
  eprint       = {1511.08458},
  timestamp    = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/OSheaN15.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@article{DBLP:journals/corr/abs-1207-0580,
  author       = {Geoffrey E. Hinton and
                  Nitish Srivastava and
                  Alex Krizhevsky and
                  Ilya Sutskever and
                  Ruslan Salakhutdinov},
  title        = {Improving neural networks by preventing co-adaptation of feature detectors},
  journal      = {CoRR},
  volume       = {abs/1207.0580},
  year         = {2012},
  url          = {http://arxiv.org/abs/1207.0580},
  eprinttype    = {arXiv},
  eprint       = {1207.0580},
  timestamp    = {Mon, 13 Aug 2018 16:46:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-1207-0580.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@misc{gonzález2018galaxydetectionidentificationusing,
      title={Galaxy detection and identification using deep learning and data augmentation}, 
      author={Roberto E. González and Roberto P. Muñoz and Cristian A. Hernández},
      year={2018},
      eprint={1809.01691},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/1809.01691}, 
}
```

```bibtex
@article{Leung_2018,
   title={Deep learning of multi-element abundances from high-resolution spectroscopic data},
   ISSN={1365-2966},
   url={http://dx.doi.org/10.1093/mnras/sty3217},
   DOI={10.1093/mnras/sty3217},
   journal={Monthly Notices of the Royal Astronomical Society},
   publisher={Oxford University Press (OUP)},
   author={Leung, Henry W and Bovy, Jo},
   year={2018},
   month=nov }
```
```bibtex
@misc{astroNN,
    author={Leung Henry and Bovy Jo},
    url={https://github.com/henrysky/astroNN},
}
```
