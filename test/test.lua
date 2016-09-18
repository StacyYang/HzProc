--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
-- Created by: Hang Zhang
-- ECE Department, Rutgers University
-- Email: zhang.hang@rutgers.edu
-- Copyright (c) 2016
--
-- Feel free to reuse and distribute this software for research or 
-- non-profit purpose, subject to the following conditions:
--  1. The code must retain the above copyright notice, this list of
--     conditions.
--  2. Original authors' names are not deleted.
--  3. The authors' names are not used to endorse or promote products
--      derived from this software 
--+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

require "hzproc"
require 'image'

-- load the image
I = image.lena():cuda()
image.display(I)

local function hzproc_testResize()
	scale = 1.0/3;
	-- generating lookuptable for scaling
	map = hzproc.Table.Resize(I:size()[3], I:size()[2], 
							I:size()[3]*scale, I:size()[2]*scale)
	map = map:cuda()
	-- mapping
	O = hzproc.Remap.Fast(I, map)
	-- display the images
	image.display(O)
end

local function hzproc_testAffine()
	-- affine transformation matrix
	mat = torch.CudaTensor({{1, 0, 0}, {.5, 1, 0}, {0, 0, 1}})
	-- affine mapping
	O = hzproc.Remap.Affine(I, mat);
	-- display the images
	image.display(O)
end

local function hzproc_testPadding()
	scale = 1.3;
	-- generating lookuptable for padding
	map = hzproc.Table.Pad(I:size()[3], I:size()[2], 
							I:size()[3]*scale, I:size()[2]*scale)
	map = map:cuda()
	-- mapping
	O = hzproc.Remap.Fast(I, map)
	-- display the images
	image.display(O)
end

local function hzproc_testCroping()
	scale  = 0.6;
	offset = (1-scale) / 2;
	-- generating lookuptable for cropping
	map = hzproc.Table.Crop(I:size()[3], I:size()[2], 
							I:size()[3]*scale, I:size()[2]*scale, 
							I:size()[3]*offset, I:size()[2]*offset)
	map = map:cuda()
	-- mapping
	O = hzproc.Remap.Fast(I, map)
	-- display the images
	image.display(O)
end

function hzproc_test()
	hzproc_testResize()	
	hzproc_testAffine()
	hzproc_testPadding()
	hzproc_testCroping()
end

hzproc_test()
