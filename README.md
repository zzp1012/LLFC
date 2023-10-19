# LLFC: Layerwise Linear Feature Connectivity

## Abstract

Code release for paper ["Going Beyond Linear Mode Connectivity: The Layerwise Linear Feature Connectivity"](https://arxiv.org/abs/2307.08286) (accepted by NeurIPS 2023). 

> Recent work has revealed many intriguing empirical phenomena in neural network training, despite the poorly understood and highly complex loss landscapes and training dynamics. One of these phenomena, Linear Mode Connectivity (LMC), has gained considerable attention due to the intriguing observation that different solutions can be connected by a linear path in the parameter space while maintaining near-constant training and test losses. In this work, we introduce a stronger notion of linear connectivity, Layerwise Linear Feature Connectivity (LLFC), which says that the feature maps of every layer in different trained networks are also linearly connected. We provide comprehensive empirical evidence for LLFC across a wide range of settings, demonstrating that whenever two trained networks satisfy LMC (via either spawning or permutation methods), they also satisfy LLFC in nearly all the layers. Furthermore, we delve deeper into the underlying factors contributing to LLFC, which reveal new insights into the spawning and permutation approaches. The study of LLFC transcends and advances our understanding of LMC by adopting a feature-learning perspective.

## Requirements

1. Make sure GPU is avaible and `CUDA>=11.0` has been installed on your computer. You can check it with
    ```bash
        nvidia-smi
    ```
2. If you use anaconda3 or miniconda, you can run following instructions to download the required packages in python.
    ```bash
        conda env create -f environment.yml
    ```

---------------------------------------------------------------------------------
Shanghai Jiao Tong University - Email@[yanjunchi@sjtu.edu.cn](yanjunchi@sjtu.edu.cn)