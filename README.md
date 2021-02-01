## Iterative Filter Adaptive Network for Single Image Defocus Deblurring &mdash; Official PyTorch Implementation
![Python 3.8.5](https://img.shields.io/badge/python-3.8.5-green.svg?style=plastic)
![PyTorch 1.6.0](https://img.shields.io/badge/PyTorch-1.6.0-green.svg?style=plastic)
![CUDA 10.1.105](https://img.shields.io/badge/CUDA-10.1.105-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=plastic)

![Teaser image](./assets/figure.jpg)

This repository contains the official PyTorch implementation of the following paper:

> **Iterative Filter Adaptive Network for Single Image Defocus Deblurring**<br>
> Junyong Lee (POSTECH), Hyeongseok Son (POSTECH), Jaesung Rim (POSTECH), Sunghyun Cho (POSTECH), Seungyong Lee (POSTECH)
> 
> http://cg.postech.ac.kr/papers/2020_CGI_JY.pdf
>
> **Abstract:** *We propose a novel end-to-end learning-based approach for single image defocus deblurring. The proposed approach is equipped with a novel Iterative Filter Adaptive Network (IFAN) that is specifically designed to handle spatially-varying large defocus blur. For adaptive handling of spatially variant blur, IFAN predicts pixel-wise deblur-ring filters, which are applied to features from an input image to generate deblurred features. For effective managing of large blur, IFAN models deblurring filters as stacks of small-sized separable filters. Predicted separable debluring filters are applied to defocused features using a novel Iterative Adaptive Convolution (IAC) layer. Besides, we propose a training scheme based on the learning of defocus disparity estimation and reblurring, which significantly boosts up the deblurring quality. We demonstrate that our method achieves state-of-the-art performance both quantitativelyand qualitatively on real-world images.*

For any inquiries, please contact [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## Resources

All material related to our paper is available via the following links:

| Link |
| :-------------- |
| [Paper PDF](https://drive.google.com/file/d/1mRVo3JefkgRd2VdJvG5M-8xWtvl60ZWg/view?usp=sharing) |
| [Supplementary Files](https://drive.google.com/file/d/1sQTGHEcko2HxoIvneyrot3bUabPrN5l1/view?usp=sharing) |
| [Checkpoint Files](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing) |


## Testing the network
1. Download pretrained weights from [here](https://drive.google.com/file/d/1Xl8cXmhlD1DjaYNcroRLMjYR3C9QplNs/view?usp=sharing).
Then, place checkpoints under `./checkpoints`.

2. Place your images under `./test`. Input images and their segment map should be placed under `./test/input` and `./test/seg_in`, respectively. Place target images and their segment map under `./test/target` and `./test/seg_tar`. 

3. To test the network, type
```bash
python test.py --dataroot [test folder path] --checkpoints_dir [ckpt path]
```

## Using pre-trained networks

## BIBTEX
If you find this code useful, please consider citing:

```
@article{Lee_2020_CTHA,
  author = {Lee, Junyong and Son, Hyeongseok and Lee, Gunhee and Lee, Jonghyeop and Cho, Sunghyun and Lee, Seungyong},
  title = {Deep Color Transfer using Histogram Analogy},
  journal = {The Visual Computer},
  volume = {36},
  number = {10},
  pages = {2129--2143},
  year = 2020,
}
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
