# Description of Comparative Algorithms

This document provides a detailed description of the comparative algorithms and related parameter settings used in this paper, covering multi-label feature selection (MFS), partial multi-label learning (PML), and partial multi-label feature selection (PMFS).

**GRRO.** GRRO jointly considers feature relevance, label relevance, and feature redundancy in a unified optimization framework to efficiently obtain globally optimal feature subsets. By optimizing global relevance and redundancy relationships, it aims to select informative features while suppressing redundant ones in multilabel learning tasks. Its parameters $\alpha_1$ and $\beta$ are tuned within $\\{0.001, 0.01, \ldots, 1000\\}$ to achieve optimal performance.

**SSFS.** SSFS captures the shared latent structure between features and labels by introducing a constrained latent structure shared term. It further incorporates graph regularization to preserve structural information, so that selected features can better reflect the intrinsic relationship between the feature space and the label space. Its parameters $\alpha$, $\beta$, and $\gamma$ are tuned within $\\{0.001, 0.01, \ldots, 1000\\}$ to achieve optimal performance.

**fPML.** fPML is a feature-induced partial multi-label learning method that exploits feature information to assist label disambiguation. It models the relationship between instances and candidate labels, and uses feature-induced evidence to estimate credible labels from partially labeled data. Its parameters are set as $\lambda_1=1$, $\lambda_2=1$, and $\lambda_3=10$.

**PML-LCom.** PML-LCom improves partial multi-label learning by using label compression to reduce label ambiguity and capture label correlations. It jointly optimizes the compressed label representation, the recovered label matrix, and the predictive model, thereby enhancing learning performance when only candidate labels are available. Its parameters are set as $\lambda_1=5$, $\lambda_2=10$, and $\lambda_3=0.1$.

**NLR.** NLR focuses on noisy label removal for partial multi-label learning. It identifies and removes false-positive labels from the candidate label set while exploiting the remaining reliable label information for model learning. This makes it suitable for PML scenarios where observed candidate labels may contain substantial noise. Its parameters are set as $\beta=10$, $\gamma=10$, and $\lambda=2$.

**PML-LRS.** PMFS-LRS integrates label disambiguation and feature selection into a unified framework by decomposing the candidate label matrix into low-rank ground-truth and sparse noisy components, while exploiting local label correlations through manifold regularization. Its parameters $\alpha$, $\beta$, and $\gamma$ are tuned within $\\{0.001, 0.01, \ldots, 1000\\}$ to achieve optimal performance.

**PML-FSLA.** PML-FSLA reconsiders feature structure information and latent space alignment in partial multi-label feature selection. It aligns the feature and label spaces in a latent representation while preserving feature structure information, so that discriminative features can be selected under ambiguous partial multi-label supervision. Its parameters are set as $\alpha=100$, $\beta=1$, and $\gamma=10$.

**DGLFS.** DGLFS performs dual-space guided global-local disambiguation for partial multi-label feature selection. It jointly exploits information from both feature and label spaces, combines global and local disambiguation mechanisms, and selects features that are more robust to noisy candidate labels. Its parameters are set as $\lambda_1=10$, $\lambda_2=0.0001$, and $\lambda_3=0.01$.

**PMFS-BAGD.** PMFS-BAGD integrates feature-label bidirectional association and adaptive label graph diffusion into a unified framework to recover reliable label distributions, capture high-order label correlations, and guide discriminative feature selection under partial multi-label learning. Its parameters are set as $\lambda_1=1$, $\lambda_2=1$, $\lambda_3=1$, and $\lambda_4=1$.

# Program Description

This repository provides the MATLAB implementation used to reproduce the experimental results of the TLSD$^2$ method. All reported results were obtained on a local computer equipped with an AMD Ryzen 9 8945HX CPU and an NVIDIA GeForce RTX 5060 Laptop GPU, using MATLAB R2022a as the running environment. The file `main.m` serves as the main entry function of the program and can be used to run the complete experimental procedure, including data loading, parameter configuration, model training, feature selection, and performance evaluation. The file `demo.m` provides the running result under the optimal parameter setting for the corresponding dataset, and can be used as a convenient example for reproducing the best reported performance on that dataset. Users may run `main.m` for the full experiment or refer to `demo.m` to quickly check the dataset-specific optimal configuration and its corresponding output.
