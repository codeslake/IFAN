# Iterative Filter Adaptive Network for Single Image Defocus Deblurring
![Python 3.8.5](https://img.shields.io/badge/python-3.8.5-green.svg?style=plastic)
![PyTorch 1.6.0](https://img.shields.io/badge/PyTorch-1.6.0-green.svg?style=plastic)
![CUDA 10.1.105](https://img.shields.io/badge/CUDA-10.1.105-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

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

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://drive.google.com/file/d/1mRVo3JefkgRd2VdJvG5M-8xWtvl60ZWg/view?usp=sharing) |
| [Supplementary Files](https://drive.google.com/file/d/1sQTGHEcko2HxoIvneyrot3bUabPrN5l1/view?usp=sharing) |
| [Checkpoint Files](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing) |

## Training/Testing the network
### Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9000 run.py \
--is_train \
--mode IFAN \
--config config_IFAN \
--trainer trainer \
--network IFAN \
--b 2 \
--th 8 \
--dl \
-dist \
```

### Testing
* Evaluating the DPDD test set
```bash
python run.py --mode IFAN --eval_mode quan --data DPDD
```
* Evaluating the RealDOF test set
```
python run.py --mode IFAN --eval_mode quan --data RealDOF
```
* Evaluating the CUHK test set
```
python run.py --mode IFAN --eval_mode quan --data CUHK
```

## Testing with pre-trained weights of CVPR2021
1. Download pretrained weights from [here](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing).
Then, unzip them under `./checkpoints`.

2. Place your images under `./test`. Input images and their segment map should be placed under `./test/input` and `./test/seg_in`, respectively. Place target images and their segment map under `./test/target` and `./test/seg_tar`. 

3. To test the network, type:
* to test the final model 
```bash
#To test our final model, 
python run.py --mode IFAN --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN.pytorch
```
* to test models used for evaluation
```
# The baseline model, 
python run.py --mode B --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/B.pytorch

# The model, D, 
python run.py --mode D --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/D.pytorch
# The model, F, 
python run.py --mode F --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/F.pytorch
# The model, FD, 
python run.py --mode FD --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/FD.pytorch
# The model, FR, 
python run.py --mode FR --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/FR.pytorch
# Our final model with N=8, 
python run.py --mode IFAN_8 --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_8.pytorch
# Our final model with N=26, 
python run.py --mode IFAN_26 --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_26.pytorch
# Our final model with N=35, 
python run.py --mode IFAN_35 --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_35.pytorch
# Our final model with N=44, 
python run.py --mode IFAN_44 --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_44.pytorch
# Our model with the FAC layer, 
python run.py --mode IFAN_FAC --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_FAC.pytorch
# Our model for dual-pixel stereo inputs, 
python run.py --mode IFAN_dual --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_dual.pytorch
# Our model trained with 16 bit images, 
python run.py --mode IFAN_16bit --eval_mode quan --data DPDD --ckpt_abs_name checkpoints/IFAN_16bit.pytorch
```

## License ##
This software is being made available under the terms in the [LICENSE](LICENSE) file.

Any exemptions to these terms requires a license from the Pohang University of Science and Technology.

## About Coupe Project ##
Project ‘COUPE’ aims to develop software that evaluates and improves the quality of images and videos based on big visual data. To achieve the goal, we extract sharpness, color, composition features from images and develop technologies for restoring and improving by using it. In addition,ersonalization technology through userreference analysis is under study.  
    
Please checkout out other Coupe repositories in our [Posgraph](https://github.com/posgraph) github organization.

## Useful Links ##
* [Coupe Library](http://coupe.postech.ac.kr/)
* [POSTECH CG Lab.](http://cg.postech.ac.kr/)
