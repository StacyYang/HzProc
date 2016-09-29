# HzProc Package Reference Manual

This doc is a quick overview and simple usage examples of [HzProc](https://github.com/zhanghang1989/hzproc).
A detailed usage example can be found in [the test script](./../test/test.lua). 

### Crop and Flip
0. **dst = hzproc.Crop.Fast(src, width, height, x1, y1, x2, y2)**
	
	Croping and scaling to target size (if necessary).
	```lua
	require "hzproc"
	require 'image'
	
	-- load the image
	I = image.lena():cuda()
	img_width  = I:size(3)
	img_height = I:size(2)
	
	-- 
	target_w, target_h = 120, 100
	size = 80
	x1, y1 = torch.random(0, img_width - size),	torch.random(0, img_height - size)
	O = hzproc.Crop.Fast(I, target_w, target_h,	x1, y1, x1+size, y1+size)
	```

0. **dst = hzproc.Crop.Bilinear(src, width, height, x1, y1, x2, y2)**

	The bilinear interpolating version of ``hzproc.Crop.Fast``.

0. **dst = hzproc.Crop.Pad(src, x1, y1, width, height, pad)**

	Croping the image with padding.

0. **hzproc.Flip.Horizon(input)**
	
	Flip the input image horizontally.

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

	The bilinear interpolating version of ``hzproc.Transform.Fast(src, mat)``.

### Remap
0. **dst = hzproc.Remap.Fast(src, table)**

	Mapping the image based on the lookup table (nearest without interpolating), a quick example:
	```lua
	-- generating lookuptable for scaling
	scale = 2.0/3;
	map = hzproc.Table.Resize(img_width, img_height, 
	                         img_width*scale, img_height*scale)
	-- mapping
	local O = hzproc.Remap.Fast(I, map)
	```
0. **dst = hzproc.Remap.Bilinear(src, table)**

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
``inw, inh, ow, oh, xoff, yoff`` are abbreviations of input width, height and output width and height, x-axis and y-axis offset.

0. **tab = hzproc.Table.Flip(width, height)**
0. **tab = hzproc.Table.Resize(inw, inh, ow, oh)**
0. **tab = hzproc.Table.Pad(inw, inh, ow, oh)**
0. **tab = hzproc.Table.Crop(inw, inh, ow, oh, xoff, yoff)**

### Get Affine Matrix
Detail about Affine Transformation, please see [Matlab Tutorial](http://www.mathworks.com/discovery/affine-transformation.html)

0. **mat = hzproc.Affine.Scale(sx, sy)**
0. **mat = hzproc.Affine.Shift(tx, ty)**
0. **mat = hzproc.Affine.Rotate(theta)**
0. **mat = hzproc.Affine.Shear(kx, ky)**

Arround center pixel

0. **mat = hzproc.Affine.RotateArround(theta, x, y)**
0. **mat = hzproc.Affine.ScaleArround(sx, sy, x, y)**
0. **mat = hzproc.Affine.ShearArround(kx, ky, x, y)**
