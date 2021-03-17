# Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking
Ning Wang, Wengang Zhou, Jie Wang, and Houqiang Li 

Accepted by **CVPR 2021 (Oral)**

This repository includes Python (PyTorch) implementation of the TrDiMP and TrSiam trackers, to appear in CVPR 2021.

![](../main/TransformerTracker.png)

## Abstract
In video object tracking, there exist rich temporal contexts among successive frames, which have been largely overlooked in existing trackers. In this work, we bridge the individual video frames and explore the temporal contexts across them via a transformer architecture for robust object tracking. Different from classic usage of the transformer in natural language processing tasks, we separate its encoder and decoder into two parallel branches and carefully design them within the Siamese-like tracking pipelines. The transformer encoder promotes the target templates via attention-based feature reinforcement, which benefits the high-quality tracking model generation. The transformer decoder propagates the tracking cues from previous templates to the current frame, which facilitates the object searching process. Our transformer-assisted tracking framework is neat and trained in an end-to-end manner. With the proposed transformer, a simple Siamese matching approach is able to outperform the current top-performing trackers. By combining our transformer with the recent discriminative tracking pipeline, our method sets several new state-of-the-art records on prevalent tracking benchmarks. 

## Tracking results and pre-trained model

**Tracking results:** the raw results of TrDiMP/TrSiam on 7 benchmarks including OTB, UAV, NFS, VOT2018, GOT-10k, TrackingNet, and LaSOT can be found [here](https://github.com/594422814/TransformerTrack/releases/download/results/Tracking_results.zip).

**Pre-trained model:** please download the [TrDiMP model](https://github.com/594422814/TransformerTrack/releases/download/model/trdimp_net.pth.tar) and put it in the ```pytracking/networks``` folder.

### Citation
If you find this work useful for your research, please consider citing our work:
```
@inproceedings{Wang_2021_Unsupervised,
    title={Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking},
    author={Wang, Ning and Zhou, Wengang and Wang, Jie and Li, Houqiang},
    booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021}
}
```

### Acknowledge
Our transformer-assisted tracker is based on [PyTracking](https://github.com/visionml/pytracking). We sincerely thank the authors Martin Danelljan and Goutam Bhat for providing this framework.

### Contact
If you have any questions, please feel free to contact wn6149@mail.ustc.edu.cn

