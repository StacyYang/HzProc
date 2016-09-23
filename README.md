#  HZPROC
Created by [Hang Zhang](http://www.hangzh.com)

HzProc is a torch data augmentation toolbox for deep learning.
The name of "HzProc" means high speed image processing, where "Hz" (hertz) 
is the unit of frequency and "Proc" is abbreviation of processing.
HZ is also the initial of the author. It contains the functions
within the following subcategories:

- [Remap](./doc/index.md#remap) remap the images based on a lookup table;
- [Affine Transformation](./doc/index.md#affine-transformation) affine transformation;
- [Combine Transforms](./doc/index.md#combine-transforms) combining different augmentation approach;
- [Get Lookup Table](./doc/index.md#get-lookup-table) getting lookup table;
- [Get Affine Matrix](./doc/index.md#get-affine-matrix) getting affine matrix;

## Introduction
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

## Install
This package relies on [torch7](https://github.com/torch/torch7) and 
[cutorch](https://github.com/torch/cutorch). Please note the package
also relies on a NVIDIA GPU compitable with CUDA 6.5 or higher version.
```bash
git clone https://github.com/zhanghang1989/hzproc
cd hzproc
luarocks make hzproc-scm-1.rockspec
```

## Quick Test
The test script relies on qlua and [image](https://github.com/torch/image) package to load and display
the images. This script is a good [usage example](https://github.com/zhanghang1989/hzproc/blob/master/test/test.lua) to get started. 
```bash
qlua test/test.lua
```

