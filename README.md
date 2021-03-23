# Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking
[Ning Wang](https://594422814.github.io), [Wengang Zhou](http://staff.ustc.edu.cn/~zhwg/index.html), [Jie Wang](https://miralab.ai/), and [Houqiang Li](http://staff.ustc.edu.cn/~lihq/English.html) 

### Accepted by **CVPR 2021 (Oral)**. [[Paper Link]](https://arxiv.org/pdf/2103.11681.pdf)

This repository includes Python (PyTorch) implementation of the TrDiMP and TrSiam trackers, to appear in CVPR 2021.

![](../main/TransformerTracker.png)

## Abstract
In video object tracking, there exist rich temporal contexts among successive frames, which have been largely overlooked in existing trackers. In this work, we bridge the individual video frames and explore the temporal contexts across them via a transformer architecture for robust object tracking. Different from classic usage of the transformer in natural language processing tasks, we separate its encoder and decoder into two parallel branches and carefully design them within the Siamese-like tracking pipelines. The transformer encoder promotes the target templates via attention-based feature reinforcement, which benefits the high-quality tracking model generation. The transformer decoder propagates the tracking cues from previous templates to the current frame, which facilitates the object searching process. Our transformer-assisted tracking framework is neat and trained in an end-to-end manner. With the proposed transformer, a simple Siamese matching approach is able to outperform the current top-performing trackers. By combining our transformer with the recent discriminative tracking pipeline, our method sets several new state-of-the-art records on prevalent tracking benchmarks. 

## Tracking Results and Pretrained Model

**Tracking results:** the raw results of TrDiMP/TrSiam on 7 benchmarks including OTB, UAV, NFS, VOT2018, GOT-10k, TrackingNet, and LaSOT can be found [here](https://github.com/594422814/TransformerTrack/releases/download/results/Tracking_results.zip).

**Pretrained model:** please download the [TrDiMP model](https://github.com/594422814/TransformerTrack/releases/download/model/trdimp_net.pth.tar) and put it in the ```pytracking/networks``` folder.

TrDiMP and TrSiam share the same model. The main difference between TrDiMP and TrSiam lies in the tracking model generation. TrSiam does not utilize the background information and simply crops the target/foreground area to generate the tracking model, which can be regarded as the initialization step of TrDiMP. 

## Environment Setup

#### Clone the GIT repository.  
```bash
git clone https://github.com/594422814/TransformerTrack.git
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

Our code is based on the PyTracking framework. For more details, please refer to [PyTracking](https://github.com/visionml/pytracking).

## Training the TrDiMP/TrSiam Model

Please refer to the [README](https://github.com/594422814/TransformerTrack/blob/main/ltr/README.md) in the ```ltr``` folder.

## Testing the TrDiMP/TrSiam Tracker

Please refer to the [README](https://github.com/594422814/TransformerTrack/blob/main/pytracking/README.md) in the ```pytracking``` folder. 
As shown in ```pytracking/README.md```, you can either use this PyTracking toolkit or GOT-10k toolkit to reproduce the tracking results.


## Citation
If you find this work useful for your research, please consider citing our work:
```
@inproceedings{Wang_2021_Transformer,
    title={Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking},
    author={Wang, Ning and Zhou, Wengang and Wang, Jie and Li, Houqiang},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

## Acknowledgment
Our transformer-assisted tracker is based on [PyTracking](https://github.com/visionml/pytracking). We sincerely thank the authors Martin Danelljan and Goutam Bhat for providing this great framework.

## Contact
If you have any questions, please feel free to contact wn6149@mail.ustc.edu.cn

