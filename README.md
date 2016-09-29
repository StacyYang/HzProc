#  HZPROC
Created by [Hang Zhang](http://www.hangzh.com)

HzProc is a torch data augmentation toolbox for deep learning.
The name of "HzProc" means high speed image processing, where "Hz" (hertz) 
is the unit of frequency and "Proc" is abbreviation of processing.
HZ is also the initial of the author. It contains the functions
within the following subcategories:

**[Quick Start Example](#quick-demo)**
* **Online** 
	- [Crop and Flip](./doc/index.md#crop-and-flip)
	- [Affine Transformation](./doc/index.md#affine-transformation) 
	- [Get Affine Matrix](./doc/index.md#get-affine-matrix) 
* **Offline**
	- [Remap](./doc/index.md#remap) 
	- [Get Lookup Table](./doc/index.md#get-lookup-table) 
	- [Combine Transforms](./doc/index.md#combine-transforms) 

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

## Quick Demo
**Demo** The [demo script](./test/demo.lua) relies on [qtlua](https://github.com/torch/qtlua) and 
[image](https://github.com/torch/image) package to load and display 
the images. The [test script](./test/test.lua) is also a good usage example. 
The documentations can be found in the [link](./doc/index.md).

```bash
qlua test/demo.lua
```
<div style="text-align:center"><img src ="./images/demo.gif" width="200" /></div>

**Fast Examples**
- online example:
```lua
function RandomCrop(size, pad)
  return function(input)
		-- avoid unnecessary opterations
    local w, h = input:size(3), input:size(2)
    if w == size and h == size then
      return input
    end
    local x1, y1 = torch.random(0, w + 2*pad - size), 
									torch.random(0, h + 2*pad - size)
    local out = hzproc.Crop.Pad(input, x1, y1, size, size, pad)
    assert(out:size(2) == size and out:size(3) == size, 'wrong crop size')
    return out
  end
end
```
- offline example:
```lua
function Resize(inw, inh, ow, oh)
	-- generating the lookup map only once during init
	local map = hzproc.Table.Resize(inw, inh, ow, oh)
  return function(input)
		-- avoid unnecessary opterations
	  local w, h = input:size(3), input:size(2)
    if (w == ow and h == oh) then
      return input
    end	
		-- mapping
		return hzproc.Remap.Bilinear(input, map)
  end
end

```


