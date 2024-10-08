# [Human4DiT: 360-degree Human Video Generation with 4D Diffusion Transformer (SIGGRAPH ASIA 2024 Journal Track)](https://human4dit.github.io)
[Ruizhi Shao*](https://dsaurus.github.io/saurus/), [Youxin Pang*](), [Zerong Zheng](http://zhengzerong.github.io/), [Jingxiang Sun](https://mrtornado24.github.io), [Yebin Liu](http://www.liuyebin.com/).

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2405.17405)

This repository contains the official implementation of ”*Human4DiT: 360-degree Human Video Generation with 4D Diffusion Transformer*“.

![Teaser Image](assets/teaser.png)

### News
* **[2024/10/08]** [Human4DiT dataset]() is available!


### TODO
- [x] Human4DiT dataset
- [ ] Human4DiT dataset preprocssing code
- [ ] Human4DiT model and inference code
- [ ] Human4DiT training code

## Human4DiT Dataset

### Dataset Structure
**Human4DiT dataset** consists of 10K monocular human videos from the internet, 5k 3D human scans captured by a dense DLSR rig and 100 4D human characters. 

For monocular human videos, we provide their downlond urls and corresponding SMPL sequences. For human scans, we provide 3D models (obj file) and the estimated SMPL model. For 4D human characters, we provide FBX model files.

### Agreement
1. The Human4DiT dataset (the "Dataset") is available for **non-commercial** research purposes only. Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The dataset may not be used for pornographic purposes or to generate pornographic material whether commercial or not. The Dataset may not be reproduced, modified and/or made available in any form to any third party without Tsinghua University’s prior written permission.

2. You agree **not to** reproduce, modified, duplicate, copy, sell, trade, resell or exploit any portion of the images and any portion of derived data in any form to any third party without Tsinghua University’s prior written permission.

3. You agree **not to** further copy, publish or distribute any portion of the Dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

4. Tsinghua University reserves the right to terminate your access to the Dataset at any time.

### Download Instructions 
The dataset is encrypted to prevent unauthorized access.

Please fill the [request form](https://docs.google.com/forms/d/e/1FAIpQLScMfdqBL3e1fLfka3THCo2Kmuf6Wzv0q-iFMshao3D3u6ZFHQ/viewform?usp=sf_link) and get the download links of Human4DiT dataset.

By requesting for the link, you acknowledge that you have read the agreement, understand it, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Dataset.

## Citation
```
@article{shao2024human4dit,
title={Human4DiT: 360-degree Human Video Generation with 4D Diffusion Transformer},
author={Shao, Ruizhi and Pang, Youxin and Zheng, Zerong and Sun, Jingxiang and Liu, Yebin},
journal={ACM Transactions on Graphics (TOG)},
volume={43},
number={6},
articleno={},
year={2024}, publisher={ACM New York, NY, USA}
}
```