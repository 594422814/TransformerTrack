# Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking
A general python framework for visual object tracking and video object segmentation, based on **PyTorch**.

### New version released!
* Code for our CVPR 2020 paper [Probabilistic Regression for Visual Tracking](https://arxiv.org/abs/2003.12565).  
* Tools for analyzing results: performance metrics, plots, tables, etc.  
* Support for multi-object tracking. Any tracker can be run in multi-object mode.    
* Support for Video Object Segmentation (**VOS**): training, datasets, evaluation, etc.  
* Code for [Learning What to Learn for Video Object Segmentation](https://arxiv.org/abs/2003.11540) will be released soon.  
* Much more...   

**Note:** Many of our changes are breaking. Integrate your extensions into the new version of PyTracking should not be difficult.
We advise to check the updated implementation and train scripts of DiMP in order to update your code.

## Highlights



Libraries for implementing and evaluating visual trackers. It includes

* All common **tracking** and **video object segmentation** datasets.  
* Scripts to **analyse** tracker performance and obtain standard performance scores.
* General building blocks, including **deep networks**, **optimization**, **feature extraction** and utilities for **correlation filter** tracking.  

### [Training Framework: LTR](ltr)
 
**LTR** (Learning Tracking Representations) is a general framework for training your visual tracking networks. It is equipped with

* All common **training datasets** for visual object tracking and segmentation.  
* Functions for data **sampling**, **processing** etc.  
* Network **modules** for visual tracking.
* And much more...


## Tracker
The toolkit contains the implementation of the following trackers.  



## [Model Zoo](MODEL_ZOO.md)
The tracker models trained using PyTracking, along with their results on standard tracking 
benchmarks are provided in the [model zoo](MODEL_ZOO.md). 


## Installation

#### Clone the GIT repository.  
```bash
git clone https://github.com/visionml/pytracking.git
```
   
#### Clone the submodules.  
In the repository directory, run the commands:  
```bash
git submodule update --init  
```  
#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path pytracking
```  
This script will also download the default networks and set-up the environment.  

**Note:** The install script has been tested on an Ubuntu 18.04 system. In case of issues, check the [detailed installation instructions](INSTALL.md). 

**Windows:** (NOT Recommended!) Check [these installation instructions](INSTALL_win.md). 

#### Let's test it!
Activate the conda environment and run the script pytracking/run_webcam.py to run ATOM using the webcam input.  
```bash
conda activate pytracking
cd pytracking
python run_webcam.py dimp dimp50    
```  



## Main Contributors

* [Martin Danelljan](https://martin-danelljan.github.io/)  
* [Goutam Bhat](https://www.vision.ee.ethz.ch/en/members/detail/407/)
