# LTR

A general PyTorch based framework for learning tracking representations. 
## Table of Contents

* [Quick Start](#quick-start)
* [Overview](#overview)
* [Trackers](#trackers)
   * [TrDiMP/TrSiam](#TrDiMP)
   * [ATOM](#ATOM)
* [Transformer in TrDiMP](#transformer-in-trdimp)
* [Training your own networks](#training-your-own-networks)

## Quick Start
The installation script will automatically generate a local configuration file  "admin/local.py". In case the file was not generated, run ```admin.environment.create_default_local_file()``` to generate it. Next, set the paths to the training workspace, 
i.e. the directory where the checkpoints will be saved. Also set the paths to the datasets you want to use. If all the dependencies have been correctly installed, you can train a network using the run_training.py script in the correct conda environment.  
```bash
conda activate pytracking
python run_training.py train_module train_name
```
Here, ```train_module``` is the sub-module inside ```train_settings``` and ```train_name``` is the name of the train setting file to be used.

For example, you can train the proposed *TrDiMP/TrSiam* tracker by running:
```bash
python run_training dimp transformer_dimp
```


## Overview
The framework consists of the following sub-modules.  
 - [actors](actors): Contains the actor classes for different trainings. The actor class is responsible for passing the input data through the network can calculating losses.  
 - [admin](admin): Includes functions for loading networks, tensorboard etc. and also contains environment settings.  
 - [dataset](dataset): Contains integration of a number of training datasets, namely [TrackingNet](https://tracking-net.org/), [GOT-10k](http://got-10k.aitestunion.com/), [LaSOT](https://cis.temple.edu/lasot/), 
 [ImageNet-VID](http://image-net.org/), [DAVIS](https://davischallenge.org), [YouTube-VOS](https://youtube-vos.org), [MS-COCO](http://cocodataset.org/#home), [SBD](http://home.bharathh.info/pubs/codes/SBD), [LVIS](https://www.lvisdataset.org), [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html), [MSRA10k](https://mmcheng.net/msra10k), and [HKU-IS](https://sites.google.com/site/ligb86/hkuis). Additionally, it includes modules to generate synthetic videos from image datasets. 
 - [data_specs](data_specs): Information about train/val splits of different datasets.   
 - [data](data): Contains functions for processing data, e.g. loading images, data augmentations, sampling frames from videos.  
 - [external](external): External libraries needed for training. Added as submodules.  
 - [models](models): Contains different layers and network definitions.  
 - [trainers](trainers): The main class which runs the training.  
 - [train_settings](train_settings): Contains settings files, specifying the training of a network.   
 
## Trackers
 The framework currently contains the training code for the following trackers.

### TrDiMP
 The following setting file can be used train the TrDiMP/TrSiam network, or to know the exact training details. 
 - [dimp.transformer_dimp](train_settings/dimp/transformer_dimp.py): The default settings used for training the TrDiMP/TrSiam model with ResNet-50 backbone. Note that TrSiam can be regarded as the initialization step of TrDiMP. Therefore, we use this trained model to perform both TrDiMP and TrSiam.

 
### ATOM
 The following setting file can be used train the ATOM network, or to know the exact training details. 
 - [bbreg.atom](train_settings/bbreg/atom_paper.py): The settings used in the paper for training the network in ATOM.
 - [bbreg.atom](train_settings/bbreg/atom.py): Newer settings used for training the network in ATOM, also utilizing the GOT10k dataset.
 - [bbreg.atom](train_settings/bbreg/atom_prob_ml.py): Settings for ATOM with the probabilistic bounding box regression proposed in [this paper](https://arxiv.org/abs/1909.12297). 
 - [bbreg.atom](train_settings/bbreg/atom_paper.py): The baseline ATOM* setting evaluated in [this paper](https://arxiv.org/abs/1909.12297).  


## Transformer in TrDiMP
 If you are familiar with DiMP tracker as well as its original codes, you can easily follow our TrDiMP/TrSiam tracker by referring to the [transformer.py](models/target_classifier/transformer.py) and [dimpnet.py](models/tracking/dimpnet.py). 

## Training your own networks
To train a custom network using the toolkit, the following components need to be specified in the train settings. For reference, see [atom.py](train_settings/bbreg/atom.py).  
- Datasets: The datasets to be used for training. A number of standard tracking datasets are already available in ```dataset``` module.  
- Processing: This function should perform the necessary post-processing of the data, e.g. cropping of target region, data augmentations etc.  
- Sampler: Determines how the frames are sampled from a video sequence to form the batches.  
- Network: The network module to be trained.  
- Objective: The training objective.  
- Actor: The trainer passes the training batch to the actor who is responsible for passing the data through the network correctly, and calculating the training loss.  
- Optimizer: Optimizer to be used, e.g. Adam.  
- Trainer: The main class which runs the epochs and saves checkpoints. 
 

 
