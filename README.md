

# Iterative Filter Adaptive Network for Single Image Defocus Deblurring
![Python 3.8.5](https://img.shields.io/badge/python-3.8.5-green.svg?style=plastic)
![PyTorch 1.6.0](https://img.shields.io/badge/PyTorch-originally%20with%201.6.0,%20works%20fine%20with%201.7.1-green.svg?style=plastic)
![CUDA 10.1.105](https://img.shields.io/badge/CUDA-10.1.105-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-blue.svg?style=plastic)

![Teaser image](./assets/figure.jpg)

This repository contains the official PyTorch implementation of the following paper:

> **[Iterative Filter Adaptive Network for Single Image Defocus Deblurring](http://cg.postech.ac.kr/papers/2020_CGI_JY.pdf)**<br>
> Junyong Lee, Hyeongseok Son, Jaesung Rim, Sunghyun Cho, Seungyong Lee, CVPR2021

If you find this code useful, please consider citing:
```
@InProceedings{Lee_2021_CVPR,
author = {Lee, Junyong and Son, Hyeongseok and Rim, Jaesung and Cho, Sunghyun and Lee, Seungyong},
title = {Iterative Filter Adaptive Network for Single Image Defocus Deblurring},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

For any inquiries, please contact [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available by following links:

| Link |
| :-------------- |
| [The main paper]() |
| [Supplementary]() |
| [Checkpoint Files](https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=0) |
| The DPDD dataset ([download](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=0)/[reference](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel)) |
| The PixelDP test set ([download](https://www.dropbox.com/s/pw7w2bpp7pc410n/PixelDP.zip?dl=0)/[reference](https://ln2.sync.com/dl/ec54aa480/b28q2xma-9xa3w5tx-ss2cv7dg-2yx935qs/view/default/10770664900008)) |
| The CUHK dataset ([download](https://www.dropbox.com/s/zxjhzuxsxh4v0cv/CUHK.zip?dl=0)/[reference](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/dataset.html)) |
| [The RealDOF test set](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=0) |
*The main paper and supplementary are provisional accepted versions. They will be updated when camera ready versions are ready*

## Training & testing of the network
*Requirements*: `pip install -r requirements.txt`
### Training
>Download and Place [DPDD dataset](https://www.dropbox.com/s/y0cc3loytfbd81h/DPDD.zip?dl=0) under `./datasets` (the offset can be modified by `config.data_offset` in `./configs/config.py`).

```bash
# multi GPU (with DistributedDataParallel) example
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
--is_train \
--mode IFAN \
--config config_IFAN \
--trainer trainer \
--network IFAN \
-b 2 \
-th 8 \
-dl \
-dist

# single GPU (with DataParallel) example
CUDA_VISIBLE_DEVICES=0 python -B run.py \
--is_train \
--mode IFAN \
--config config_IFAN \
--trainer trainer \
--network IFAN \
-b 8 \
-th 8 \
-dl
```
* options
    * `--is_train`: If it is specified, `run.py` will train the network.  
    * `--mode`: The name of a training mode. The logging folder named with the `mode` will be created under `./logs/Defocus_Deblurring/[mode]`. 
    * `--config`: The name of a config file located as in `./config/[config].py`.
    * `--trainer`: The name of a trainer file located as in `./models/trainers/[trainer].py`.
    * `--network`: The name of a network file located as in `./models/archs/[network].py`.
    * `-b`: The batch size. With multi GPUs, (`DistributedDataParallel`), the total batch size will be, `nproc_per_node * b`.
    * `-th`: The number of threads (`num_workers`) for the data loader (defined in `./models/baseModel`).
    * `-dl`: The option whether to delete logs under `mode` (i.e., `./logs/Defocus_Deblurring/[mode]/*`). The option works only when `--is_train` is specified.
    * `-r`: Resume training with the specified epoch # (e.g., `-r 100`). Note that `-dl` should not be specified with this option.
    * `-dist`: whether to use `DistributedDataParallel`.
* logs
    * The root offset of logs is `./logs`. It can be changed by `config.log_offset` in `./config/config.py`. In `./logs/Defocus_Deblurring/[mode]/`, checkpoints, resume states, config, scalar logs for tensorboard and testing results will be saved.
    * Currently, the logs will be saved for every 4 epochs, which can be reset by `config.write_ckpt_every_epoch` in `./config/config.py`.

### Testing
```bash
python run.py --mode [MODE] --data [DATASET]
# e.g., python run.py --mode IFAN --data DPDD
```
* options
    * `--mode`: The name of the training mode that you want to test.
    * `--data`: The name of a dataset for evaluation. We have `DPDD, RealDOF, CUHK, PixelDP, any`, and their path can be modified by the function `set_eval_path(..)` in `./configs/config.py`.
    * `-ckpt_name`: Loads the checkpoint with the name of the checkpoint under `./logs/Defocus_Deblurring/[mode]/checkpoint/train/epoch/ckpt/` (e.g., `python run.py --mode IFAN --data DPDD --ckpt_name IFAN_00000.pytorch`).
    * `-ckpt_abs_name`. Loads the checkpoint of the absolute path (e.g., `python run.py --mode IFAN --data DPDD --ckpt_abs_name ./checkpoints/IFAN.pytorch`).
    * `-ckpt_epoch`: Loads the checkpoint of the specified epoch (e.g., `python run.py --mode IFAN --data DPDD --ckpt_epoch 0`). 
    * `-ckpt_sc`: Loads the checkpoint with the best validation score (e.g., `python run.py --mode IFAN --data DPDD --ckpt_sc`)    
* results
    * Testing results will be saved in `./logs/Defocus_Deblurring/[mode]/result/quanti_quali/[mode]_[epoch]/data`.

## Testing with pre-trained weights of CVPR2021
> Download pretrained weights from [here](https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=0). Then, unzip them under `./checkpoints`.

> Download and place test sets ([DPDD](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=0), [PixelDP](https://www.dropbox.com/s/pw7w2bpp7pc410n/PixelDP.zip?dl=0), [CUHK](https://www.dropbox.com/s/zxjhzuxsxh4v0cv/CUHK.zip?dl=0) and [RealDOF](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=0)) under `./datasets` (the offset can be modified by `config.EVAL.test_offset` in `./configs/config.py`).

1. To test the final model,
    ```bash
    # Our final model 
    python run.py --mode IFAN --network IFAN --config config_IFAN --data DPDD --ckpt_abs_name checkpoints/IFAN.pytorch
    ```
    * `--data`: The name of a dataset for evaluation. We have `DPDD, RealDOF, CUHK, PixelDP, any`, and their path can be modified by the function `set_eval_path(..)` in `./configs/config.py`. `--data any` is for testing models with any images, which should be placed under the folder `./datasets/any`. 


2. To test our models used for evaluations in the paper,
    ```bash
    ## Table 4 in the main paper
    # Our final model with N=8
    python run.py --mode IFAN_8 --network IFAN --config config_IFAN_8 --data DPDD --ckpt_abs_name checkpoints/IFAN_8.pytorch

    # Our final model with N=26
    python run.py --mode IFAN_26 --network IFAN --config config_IFAN_26 --data DPDD --ckpt_abs_name checkpoints/IFAN_26.pytorch

    # Our final model with N=35
    python run.py --mode IFAN_35 --network IFAN --config config_IFAN_35 --data DPDD --ckpt_abs_name checkpoints/IFAN_35.pytorch

    # Our final model with N=44
    python run.py --mode IFAN_44 --network IFAN --config config_IFAN_44 --data DPDD --ckpt_abs_name checkpoints/IFAN_44.pytorch

    ## Table 1 in the supplementary material
    # Our model trained with 16 bit images
    python run.py --mode IFAN_16bit --network IFAN --config config_IFAN_16bit --data DPDD --ckpt_abs_name checkpoints/IFAN_16bit.pytorch

    ## Table 2 in the supplementary material
    # Our model for dual-pixel stereo inputs 
    python run.py --mode IFAN_dual --network IFAN_dual --config config_IFAN --data DPDD --ckpt_abs_name checkpoints/IFAN_dual.pytorch
    ```


## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using them. In addition, personalization technology through user reference analysis is under study.  
    
Please checkout other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)
