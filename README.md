## IFAN: Iterative Filter Adaptive Network for Single Image Defocus Deblurring<br><sub>Official PyTorch Implementation of the CVPR 2021 Paper</sub><br><sub>[Project](https://junyonglee.me/projects/IFAN) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf) | [arXiv](https://arxiv.org/pdf/2108.13610.pdf) | [Supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Lee_Iterative_Filter_Adaptive_CVPR_2021_supplemental.pdf) | [Poster](https://www.dropbox.com/s/8kpa61f1nnv0ato/IFAN_Poster.pdf?raw=1) | [Slide](https://www.dropbox.com/s/kpp6mxxxl5lah1n/IFAN_slides.pdf?raw=1)</sub><br><sub><sub>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DmazbJPUlx4MF9-Z9llvddlywxlLWxsX?usp=sharing) [![License CC BY-NC](https://img.shields.io/badge/Replicate-Open_in_Replicate-blue.svg?style=flat)](https://replicate.ai/codeslake/ifan-defocus-deblur) [![License CC BY-NC](https://img.shields.io/badge/Anvil-Open_in_Anvil_(fastest,_but_may_be_offline)-blue.svg?style=flat)](https://YJ5YKNVB7BY5PN7Y.anvil.app/KNK4MOE27FW3VZNDQUNHAJAY)</sub></sub>

This repo contains training and evaluation code for the following paper:

> [**Iterative Filter Adaptive Network for Single Image Defocus Deblurring**](https://junyonglee.me/projects/IFAN)<br>
> [Junyong Lee](https://junyonglee.me), [Hyeongseok Son](https://sites.google.com/site/sonhspostech/), [Jaesung Rim](https://github.com/rimchang), [Sunghyun Cho](https://www.scho.pe.kr/), and [Seungyong Lee](http://cg.postech.ac.kr/leesy/)<br>
>  POSTECH <br>
> *IEEE Computer Vision and Pattern Recognition (**CVPR**) 2021*<br>


<p align="left">
  <a href="https://junyonglee.me/projects/IFAN">
    <img width=85% src="./assets/teaser.gif"/>
  </a><br>
</p>

## Getting Started
### Prerequisites

*Tested environment*

![Ubuntu](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%20&%201.8.0%20&%201.9.0%20&%201.10.2%20&%201.11.0-green.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20&%2011.1%20&%2011.3-green.svg?style=plastic)

#### 1. Environment setup
* Option 1. install from scratch
    ```bash
    $ git clone https://github.com/codeslake/IFAN.git
    $ cd IFAN

    ## for CUDA10.2
    $ conda create -y --name IFAN python=3.6 && conda activate IFAN
    $ sh install_CUDA10.2.sh

    ## for CUDA11.1 or CUDA11.3
    $ conda create -y --name IFAN python=3.8 && conda activate IFAN
    # CUDA11.1
    $ sh install_CUDA11.1.sh

    # CUDA11.3 (for amp)
    $ sh install_CUDA11.3.sh
    ```

* Option 2. docker
    ```bash
    $ nvidia-docker run --privileged --gpus=all -it --name IFAN --rm codeslake/IFAN:CVPR2021 /bin/zsh
    $ git clone https://github.com/codeslake/IFAN.git
    $ cd IFAN

    # for CUDA10.2
    $ coda activate IFAN_CUDA10.2

    # for CUDA11.1
    $ coda activate IFAN_CUDA11.1

    # for CUDA11.3 (for amp)
    $ coda activate IFAN_CUDA11.3
    ```


#### 2. Datasets
Download and unzip datasets under `[DATASET_ROOT]`:
* DPDD dataset: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%2123628&authkey=!AJTXeSG1QzF1q5A) | [Dropbox](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=1)
* PixelDP test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21487&authkey=!AAnKOuOL7yHLM94) | [Dropbox](https://www.dropbox.com/s/pw7w2bpp7pc410n/PixelDP.zip?dl=1)
* CUHK test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21486&authkey=!ABnNaUEHo63ZPKI) | [Dropbox](https://www.dropbox.com/s/zxjhzuxsxh4v0cv/CUHK.zip?dl=1)
* RealDOF test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21511&authkey=!AJu9MY_YydEc-aU) | [Dropbox](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1)

```
[DATASET_ROOT]
 ├── DPDD
 ├── PixelDP
 ├── CUHK
 └── RealDOF
```
> `[DATASET_ROOT]` can be modified with [`config.data_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L48-L49) in `./configs/config.py`.

#### 3. Pre-trained models
Download and unzip pretrained weights ([OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21495&authkey=!AKmlSKrsKPa9cQ0) | [Dropbox](https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=1)) under `./ckpt/`:

```
.
├── ...
├── ./ckpt
│   ├── IFAN.pytorch
│   ├── ...
│   └── IFAN_dual.pytorch
└── ...
```

## Testing models of CVPR 2021

***Important Note:** There is a mistake in the [IAC layer](https://github.com/codeslake/IFAN/blob/main/models/IAC.py#L36). The current code uses isotropic kernels in which the two separable kernels are exactly the same (per iteration).
However, the current IFAN still works, as defocus blur is usually in the isotropic shape, which the current IAC layer can handle with the isotropic kernels.*

*If you are to train the code from scratch, please update the [IAC layer](https://github.com/codeslake/IFAN/blob/main/models/IAC.py#L36) to use `kernel1` and `kernel2` as commented in the code. I will update the experimental results based on the corrected IAC layer.*

```shell
## Table 2 in the main paper
# Our final model used for comparison
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN --network IFAN --config config_IFAN --data DPDD --ckpt_abs_name ckpt/IFAN.pytorch --data_offset /data_offset --output_offset ./output

## Table 4 in the main paper
# Our final model with N=8
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_8 --network IFAN --config config_IFAN_8 --data DPDD --ckpt_abs_name ckpt/IFAN_8.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=26
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_26 --network IFAN --config config_IFAN_26 --data DPDD --ckpt_abs_name ckpt/IFAN_26.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=35
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_35 --network IFAN --config config_IFAN_35 --data DPDD --ckpt_abs_name ckpt/IFAN_35.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=44
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_44 --network IFAN --config config_IFAN_44 --data DPDD --ckpt_abs_name ckpt/IFAN_44.pytorch --data_offset /data_offset --output_offset ./output

## Table 1 in the supplementary material
# Our model trained with 16 bit images
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_16bit --network IFAN --config config_IFAN_16bit --data DPDD --ckpt_abs_name ckpt/IFAN_16bit.pytorch --data_offset /data_offset --output_offset ./output

## Table 2 in the supplementary material
# Our model taking dual-pixel stereo images as an input
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_dual --network IFAN_dual --config config_IFAN --data DPDD --ckpt_abs_name ckpt/IFAN_dual.pytorch --data_offset /data_offset --output_offset ./output
```

> Testing results will be saved in `[LOG_ROOT]/IFAN_CVPR2021/[mode]/result/quanti_quali/[mode]_[epoch]/[data]/`.

> `[LOG_ROOT]` can be modified with [`config.log_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L65) in `./configs/config.py`.

#### Options
* `--data`: The name of a dataset to evaluate. `DPDD` | `RealDOF` | `CUHK` | `PixelDP` | `random`. Default: `DPDD`
    * The folder structure can be modified in the function [`set_eval_path(..)`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L114-L139) in `./configs/config.py`.
    * `random` is for testing models with any images, which should be placed as `[DATASET_ROOT]/random/*.[jpg|png]`.

## Wiki
* [Logging](https://github.com/codeslake/IFAN/wiki/Log-Details)
* [Training and testing details](https://github.com/codeslake/IFAN/wiki/Training-&-Testing-Details).

## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## License
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=flat)<br>
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## Citation
If you find this code useful, please consider citing:

```
@InProceedings{Lee2021IFAN,
    author    = {Junyong Lee and Hyeongseok Son and Jaesung Rim and Sunghyun Cho and Seungyong Lee},
    title     = {Iterative Filter Adaptive Network for Single Image Defocus Deblurring},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021}
}
```

