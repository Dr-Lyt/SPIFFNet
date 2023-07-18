# SPIFFNet
SPIFFNet Codes
# Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network for Remote Sensing Image Super-Resolution
Official Pytorch implementation of the paper "Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network for Remote Sensing Image Super-Resolution".  

Abstract
Remote sensing image super-resolution (RSISR) plays a vital role in enhancing spatial detials and improving the quality of satellite imagery. Recently, Transformer-based models have shown competitive performance in RSISR. To mitigate the quadratic computational complexity resulting from global self-attention, various methods constrain attention to a local window, enhancing its efficiency. Consequently, the receptive fields in a single attention layer are inadequate, leading to insufficient context modeling. Furthermore, while most transform-based approaches reuse shallow features through skip connections, relying solely on these connections treats shallow and deep features equally, impeding the model's ability to characterize them. To address these issues, we propose a novel transformer architecture called Cross-Spatial Pixel Integration and Cross-Stage Feature Fusion Based Transformer Network (SPIFFNet) for RSISR. Our proposed model effectively enhances global cognition and understanding of the entire image, facilitating efficient integration of features cross-stages. The model incorporates cross-spatial pixel integration attention (CSPIA) to introduce contextual information into a local window, while cross-stage feature fusion attention (CSFFA) adaptively fuses features from the previous stage to improve feature expression in line with the requirements of the current stage. We conducted comprehensive experiments on multiple benchmark datasets, demonstrating the superior performance of our proposed SPIFFNet in terms of both quantitative metrics and visual quality when compared to state-of-the-art methods.

## Requirements
- Python 3.6+
- Pytorch>=1.6
- torchvision>=0.7.0
- einops
- matplotlib
- cv2
- scipy
- tqdm
- scikit


## Installation
Clone or download this code and install aforementioned requirements 
```
cd codes
```

## Train
Download the UCMerced dataset[[Baidu Drive](https://pan.baidu.com/s/1ijFUcLozP2wiHg14VBFYWw),password:terr][[Google Drive](https://drive.google.com/file/d/12pmtffUEAhbEAIn_pit8FxwcdNk4Bgjg/view)]and AID dataset[[Baidu Drive](https://pan.baidu.com/s/1Cf-J_YdcCB2avPEUZNBoCA),password:id1n][[Google Drive](https://drive.google.com/file/d/1d_Wq_U8DW-dOC3etvF4bbbWMOEqtZwF7/view)], they have been split them into train/val/test data, where the original images would be taken as the HR references and the corresponding LR images are generated by bicubic down-sample. 
```
# x4
python demo_train.py --model=SPIFFNET --dataset=UCMerced --scale=4 --patch_size=192 --ext=img --save=SPIFFNETx4_UCMerced
# x3
python demo_train.py --model=SPIFFNET--dataset=UCMerced --scale=3 --patch_size=144 --ext=img --save=SPIFFNETx3_UCMerced
# x2
python demo_train.py --model=SPIFFNET --dataset=UCMerced --scale=2 --patch_size=96 --ext=img --save=SPIFFNETx2_UCMerced
```
Download the trained model in this paper [Baidu Drive](https://pan.baidu.com/s/1qN-P-XyZoScvIJWx0fi0hA),password:zheu]. 

The train/val data pathes are set in [data/__init__.py](codes/data/__init__.py) 

## Test 
The test data path and the save path can be edited in [demo_deploy.py](codes/demo_deploy.py)

```
# x4
python demo_deploy.py --model=SPIFFNET --scale=4 --patch_size=256 --test_block=True
# x3
python demo_deploy.py --model=SPIFFNET --scale=3 --patch_size=256 --test_block=True
# x2
python demo_deploy.py --model=SPIFFNET --scale=2 --patch_size=256 --test_block=True
```

## Evaluation 
Compute the evaluated results in term of PSNR and SSIM, where the SR/HR paths can be edited in [calculate_PSNR_SSIM.py](codes/metric_scripts/calculate_PSNR_SSIM.py)

```
cd metric_scripts 
python calculate_PSNR_SSIM.py
```


## Acknowledgements 
This code is built on [HSENET (Pytorch)](https://github.com/Shaosifan/HSENet) and [TRANSENET (Pytorch)](https://github.com/Shaosifan/TransENet). 
The LAM results in this paper is tested on [LAM_DEMO (Pytorch)](https://github.com/X-Lowlevel-Vision/LAM_Demo). 
We thank the authors for sharing the codes.  


