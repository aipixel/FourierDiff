# FourierDiff
Official implement of [Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring](https://openaccess.thecvf.com/content/CVPR2024/papers/Lv_Fourier_Priors-Guided_Diffusion_for_Zero-Shot_Joint_Low-Light_Enhancement_and_Deblurring_CVPR_2024_paper.pdf)

## Installation
### Environment
```
conda env create --file environment.yml
conda activate FourierDiff
```
### Pre-Trained Models
download this [model](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)(from [guided-diffusion](https://github.com/openai/guided-diffusion)) and put it into `FourierDiff/exp/logs/imagenet/`.
```
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```
### Quick Start
The input image should be in `FourierDiff/exp/datasets/test/low`.
The results should be in `FourierDiff/exp/image_samples/output`.
```
 python main.py --config llve.yml --path_y test -i output
```
## TODO

- [x] low-light enhancement branch
- [ ] deblurring branch
## Citation
```
@inproceedings{lv2024fourier,
  title={Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light Enhancement and Deblurring},
  author={Lv, Xiaoqian and Zhang, Shengping and Wang, Chenyang and Zheng, Yichen and Zhong, Bineng and Li, Chongyi and Nie, Liqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25378--25388},
  year={2024}
}
```
