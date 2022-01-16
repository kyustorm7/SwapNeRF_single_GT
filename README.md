# GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields

If you find our code or paper useful, please cite as

    @inproceedings{GIRAFFE,
        title = {GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields},
        author = {Niemeyer, Michael and Geiger, Andreas},
        booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## TL; DR - Quick Start

![Rotating Cars](gfx/rotation_cars.gif)
![Tranlation Horizontal Cars](gfx/tr_h_cars.gif)
![Tranlation Horizontal Cars](gfx/tr_d_cars.gif)

First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `giraffe` using
```
conda env create -f environment.yml
conda activate giraffe
```

You can now test our code on the provided pre-trained models.
For example, simply run
```
python render.py configs/256res/cars_256_pretrained.yaml
```
This script should create a model output folder `out/cars256_pretrained`.
The animations are then saved to the respective subfolders in `out/cars256_pretrained/rendering`.

## Usage

### Datasets

## Evaluation
FID for reconstruction | shape swap | appearance swap | camera swap
```
python eval/eval.py configs/default.yaml
```

FID, PSNR, SSIM, LPIPS for reconstruction
```
python eval/eval_recon.py configs/default.yaml
```