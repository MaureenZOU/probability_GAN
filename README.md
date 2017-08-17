# probability_GAN
> This work was introduced in the summer research program supervised by Prof. Vijayakumar Bhagavatula hosted by Carnegie Mellon University after I finish my second year study in computer science department.

Currently, generative adversarial net is widely used in domain adaptation, image to image translation, adversarial training and etc. A bunch of different GANs are proposed to solve these problems, most of them proposed a new loss function and experiment on image datasets. But nearly none of them explain GAN back to the probability view. In this project, I explore the insight of GAN, simGAN and cycleGAN in distribution level.

## Installation

Pytorch with cpu Version

## Usage

mixGau-simGAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using simGAN [1]

mixGau_GAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using GAN [2]

mixGau_cycleGAN.py    generated mixture Gaussian data from mixture Gaussian Distribution using cycleGAN [3]

uniMix_GAN.py     generated mixture Gaussian data from uniform Distribution using GAN

uniNor_cycleGAN.py     generated mixture Gaussian data from uniform Distribution using GAN

```sh
python uniMix_GAN.py

```

## Experiment Result

`GAN mixGaussian 2 mixGaussian` https://youtu.be/2pEcTiFSMLw

`GAN uniform 2 Gaussian` https://youtu.be/Sq20c6R_XFw

`cycleGAN mixGaussian 2 mixGaussian` https://youtu.be/wvZUGTKfoLc

Example output of a single frame

![alt tag](https://raw.githubusercontent.com/MaureenZOU/probability_GAN/master/GAN.png)

## Citation

1. Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
2. Shrivastava, A., Pfister, T., Tuzel, O., Susskind, J., Wang, W., & Webb, R. (2016). Learning from simulated and unsupervised images through adversarial training. arXiv preprint arXiv:1612.07828.
3. Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593.

