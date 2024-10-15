# A Code Repository for Deep Learning (DL)-Based Motor Imagery (MI) EEG Classification

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.7.9%20|%203.9.7-EE4C2C?logo=python&style=for-the-badge" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.6.0-EE4C2C?logo=pytorch&style=for-the-badge" /></a>
      <a href= "https://www.tensorflow.org/">
      <img src=https://img.shields.io/badge/TensorFlow-2.7.0-FF6F00?logo=tensorflow&style=for-the-badge>
</a>
      <a href= "https://www.tensorflow.org/">
      <img src=https://img.shields.io/badge/Anaconda-4.10.1-44A833?logo=anaconda&style=for-the-badge>
</a>
      <a href= "https://opensource.org/license/mit">
      <img src=https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)>
</p>

This code repository collects the available source code of the representative DL-based models for classifying MI-EEG signals and runs a leaderboard table to fairly compare these models. The provided shell scripts can conveniently evaluate representative models on public MI-EEG datasets. This repository aims to help researchers learn about current state-of-the-art models and evaluate their proposed models quickly. More summarization and discussions about the representative models can be found in our survey paper [(Link)](https://www.sciencedirect.com/science/article/pii/S093336572300252X). This repository is updated regularly to contain the latest available MI-EEG decoding models.

Table of contents
===

<!--ts-->
  * [➤ Installation](#installation)
  * [➤ Usage](#usage)
  * [➤ Representative Models and Their Source Code](#Models-SourceCode)
  * [➤ Citations](#citations)
<!--te-->

<a  id="installation"></a>
Installation
===
```
git clone https://github.com/Henrywang621/DL-based-MI-EEG-models.git
cd DL-based-MI-EEG-models
conda env create -f tf-gpu.yml
conda env create -f torch37b6.yml
conda env create -f torch37c.yml
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
conda init
source ~/.bashrc
```

<a  id="usage"></a>
Usage
===
If you want to evaluate the collected models on the BCI IV 2a dataset, please use the commands below.
```
cd BCIIV2a_CrossSubjs
chmod +x ./train.sh
sh train.sh
```

<a  id="Models-SourceCode"></a>
## Representative Models and Their Corresponding Source Code

| methods | title | author |  year | source code |  
| ------ | ------ | ------ | ------ | ------ |
| Mixed LSTM/1DConv | Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks. [[Paper]](https://arxiv.org/abs/1511.06448) | Bashivan et al. | 2016 | [Code](https://github.com/pbashivan/EEGLearn) | 
| Shallow ConvNet |  Deep learning with convolutional neural networks for EEG decoding and visualization. [[Paper]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730) | Tibor et al. | 2017 | [Code](https://github.com/braindecode/braindecode) |
| Deep ConvNet | Deep learning with convolutional neural networks for EEG decoding and visualization. [[Paper]](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)| Tibor et al. | 2017 |[Code](https://github.com/braindecode/braindecode) |  
| EEGNet | EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. [[Paper]](https://arxiv.org/abs/1611.08024) | Lawhern et al. | 2018 | [Code](https://github.com/vlawhern/arl-eegmodels) |  
| An end-to-end model | An end-to-end deep learning approach to MI-EEG signal classification for BCIs. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0957417418305359) | Dose et al. | 2018 | [Code](https://github.com/hauke-d/cnn-eeg) |
| Cascade model | Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11496) | Zhang et al. | 2018 | [Code](https://github.com/dalinzhang/Cascade-Parallel) |  
| Parallel model | Cascade and Parallel Convolutional Recurrent Neural Networks on EEG-based Intention Recognition for Brain Computer Interface. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11496)| Zhang et al. | 2018 | [Code](https://github.com/dalinzhang/Cascade-Parallel) |  
| A LSTM model | Validating Deep Neural Networks for Online Decoding of Motor Imagery Movements from EEG Signals. [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/11496) | Tayeb et al. | 2018 | [Code](https://github.com/gumpy-bci/gumpy-deeplearning) |
| pCNN | Validating Deep Neural Networks for Online Decoding of Motor Imagery Movements from EEG Signals. [[Paper]](https://pubmed.ncbi.nlm.nih.gov/30626132/) | Tayeb et al. | 2019 | [Code](https://github.com/gumpy-bci/gumpy-deeplearning) |
| EEGNet fusion | Fusion Convolutional Neural Network for Cross-Subject EEG Motor Imagery Classification. [[Paper]](https://www.mdpi.com/2073-431X/9/3/72) | Roots et al. | 2020 | [Code](https://github.com/rootskar/EEGMotorImagery) |  
| C-LSTM | Data augmentation for self-paced motor imagery classification with C-LSTM. [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/ab57c0)]| Freer et al. | 2020 | [Code](https://github.com/dfreer15/DeepEEGDataAugmentation) |
| GCRAM | Motor Imagery Classification via Temporal Attention Cues of Graph Embedded EEG Signals. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8961150) | Zhang et al. | 2020 | [Code](https://github.com/dalinzhang/GCRAM) |  
| TS-SEFFNet | A Temporal-Spectral-Based Squeeze-and-Excitation Feature Fusion Network for Motor Imagery EEG Decoding. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9495768) | Li et al. | 2021 | [Code](https://github.com/LianghuiGuo/TS-SEFFNet) |  
| MIN2Net | MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9658165)| Phairot et al. | 2022 | [Code](https://github.com/IoBT-VISTEC/MIN2Net) |
| EEG-Transformer | Transformer based Spatial-Temporal Feature Learning for EEG Decoding. [[Paper]](https://arxiv.org/abs/2106.11170) | Song et al. | 2022 | [Code](https://github.com/eeyhsong/EEG-Transformer) |

<a  id="citations"></a>
Citations
===

```bibtex
@article{WANG2024102738,
title = {An in-depth survey on Deep Learning-based Motor Imagery Electroencephalogram (EEG) classification},
journal = {Artificial Intelligence in Medicine},
volume = {147},
pages = {102738},
year = {2024},
issn = {0933-3657},
doi = {https://doi.org/10.1016/j.artmed.2023.102738},
url = {https://www.sciencedirect.com/science/article/pii/S093336572300252X},
author = {Xianheng Wang and Veronica Liesaputra and Zhaobin Liu and Yi Wang and Zhiyi Huang},
```

