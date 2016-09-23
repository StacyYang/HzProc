# HzProc Package Reference Manual

This doc is a quick overview and simple usage examples of [HzProc](https://github.com/zhanghang1989/hzproc).
A detailed usage example can be found in [the test script](https://github.com/zhanghang1989/hzproc/blob/master/test/test.lua). 

### Remap
0. **dst = hzproc.Remap.Fast(src, table)**

	Mapping the image based on the lookup table (nearest without interpolating), a quick example:
```lua
require "hzproc"
require 'image'

-- load the image
I = image.lena():cuda()
img_width  = I:size()[3] 
img_height = I:size()[2] 

-- generating lookuptable for scaling
scale = 2.0/3;
map = hzproc.Table.Resize(img_width, img_height, 
                         img_width*scale, img_height*scale)
-- mapping
local O = hzproc.Remap.Bilinear(I, map)
```
0. **dst = hzproc.Remap.Bilinear(src, table)**

	The bilinear interpolating version of ``hzproc.Remap.Fast``.

### Affine Transformation
0. **dst = hzproc.Transform.Fast(src, mat)**

	Transform the image based on an affine matrix
```lua
-- affine transformation matrix
mat = torch.CudaTensor({{1,0,0},{0.3,1,0},{0,0,1}})
-- affine mapping
O = hzproc.Transform.Fast(I, mat)
```
0. **dst = hzproc.Transform.Bilinear(src, mat)**

	The bilinear interpolating version of ``hzproc.Remap.Fast``.

### Combine Transforms
0. **tab = hzproc.Transform.ToTable(mat)**

	converting the affine matrix to lookup table 
```lua
mat = torch.CudaTensor({{1,0,0},{0.3,1,0},{0,0,1}})
tab = hzproc.Transform.ToTable(mat, img_width, img_height)
```
0. **tab = hzproc.Remap.Combine(tab1, tab2)**

	combining two lookup tables
```lua
map = hzproc.Remap.Combine(map1, map2)
```

### Get Lookup Table
``inw, inh, ow, oh, xoff, yoff``abbreviation of input width, height and output width and height, x-axis and y-axis offset.
0. **tab = hzproc.Table.Resize(inw, inh, ow, oh)**
0. **tab = hzproc.Table.Pad(inw, inh, ow, oh)**
0. **tab = hzproc.Table.Crop(inw, inh, ow, oh, xoff, yoff)**

### Get Affine Matrix
Detail about Affine Transformation, please see [Matlab Tutorial](http://www.mathworks.com/discovery/affine-transformation.html)

0. **mat = hzproc.Affine.Scale(sx, sy)**
0. **mat = hzproc.Affine.Shift(tx, ty)**
0. **mat = hzproc.Affine.Rotate(theta)**
0. **mat = hzproc.Affine.Shear(kx, ky)**

