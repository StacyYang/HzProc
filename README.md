##  HZPROC
Created by [Hang Zhang](http://www.hangzh.com)

HzProc is a torch data augmentation toolbox for deep learning.
The name of "HzProc" means high speed image processing, where "Hz" (hertz) 
is the unit of frequency and "Proc" is abbreviation of processing.
HZ is also the initial of the author. 

### Introduction
Deep learning is known as one of the most popular methods in Computer 
Vision area today. 
It has achieved superior result in many competitions such as ImageNet and 
Coco. Deep Neural Network optimizes millions of the parameters during the 
training. 
However, its application on real life problem is typically limited by the 
dataset size. A sophisticated data augmentation strategy can be the 
key to improve the results and avoid overfitting. But bad 
implementation can easily become the computational bottleneck.

HzProc addresses this issue, which transfers different data 
augmentation strategies into two simple sub-problems: 
affine transformation and lookup table remapping. 
Therefore, we can generate affine transformation 
matrix or lookup table offine (every epoch), and augment the image data 
online (taking constant time). 
HzProc is an open source toolbox, which has a highly optimized CUDA 
backend. 

### Install
This package relies on [torch7](https://github.com/torch/torch7) and 
[cutorch](https://github.com/torch/cutorch). Please note the package
also relies on a NVIDIA GPU compitable with CUDA 6.5 or higher version.
```bash
git clone https://github.com/zhanghang1989/hzproc
cd hzproc
luarocks make hzproc-scm-1.rockspec
```

### Quick Test
The test script relies on qlua and [image](https://github.com/torch/image) package to load and display
the images. This script is a good [usage example](https://github.com/zhanghang1989/hzproc/blob/master/test/test.lua) to get started. 
```bash
qlua test/test.lua
```

### Usage
This package transfer data augmentation into two sub-problems: 1. lookup table remapping, 2. affine transformation. 
- **Remap**
	0. dst = hzproc.Remap.Fast(src, table)
	1. dst = hzproc.Remap.Affine(src, mat)
- **Get Lookup Table**
	0. res = hzproc.Table.Resize(inw, inh, ow, oh)
	1. res = hzproc.Table.Pad(inw, inh, ow, oh)
	2. res = hzproc.Table.Crop(inw, inh, ow, oh, xoff, yoff)
- **Get Affine Matrix**
(Detail about Affine Transformation, please see [Matlab Tutorial](http://www.mathworks.com/discovery/affine-transformation.html))
	0. res = hzproc.Affine.Scale(sx, sy)

