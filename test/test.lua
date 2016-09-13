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

local function hzproc_testScale()
	scale = 1.0/3;
	-- generating lookuptable for scaling
	map = hzproc.GetResizeTable(I:size()[2], I:size()[2], 
							I:size()[2]*scale, I:size()[2]*scale)
	map = map:cuda()
	-- mapping
	O = hzproc.Remap(I, map)
	-- display the images
	image.display(O)
end

local function hzproc_testAffine()
	-- affine transformation matrix
	mat = torch.Tensor({{1, 0, 0}, {.5, 1, 0}, {0, 0, 1}})
	mat = mat:cuda()
	-- affine mapping
	O = hzproc.AffineMap(I, mat);
	-- display the images
	image.display(O)
end

function hzproc_test()
	hzproc_testScale()	
	hzproc_testAffine()
end

hzproc_test()
