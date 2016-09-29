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

local t = require 'hzproc.online'
require 'qtwidget'
require 'qttorch'
require 'image'

-- The following color jittering params for demo are taken from 
-- fb.resnet.torch
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}
local pca = {
  eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
  eigvec = torch.Tensor{
    { -0.5675,  0.7192,  0.4009 },
    { -0.5808, -0.0045, -0.8140 },
    { -0.5836, -0.6948,  0.4203 },
  },
}

local function convert()
	return t.Compose{
		t.Resize(512,512,256,256),
    --t.Warp(0.4, 20, 2, 1.25),
    t.RandomCrop(224, 100),
    t.ColorJitter({
    	brightness = 0.4,
    	contrast = 0.4,
    	saturation = 0.4,
    }),
    t.Lighting(0.1, pca.eigval, pca.eigvec),
    t.ColorNormalize(meanstd),
    t.HorizontalFlip(0.5),
	}
end

local function sleep(n)
  local t = os.clock() + n
  while os.clock() < t do end
end

local function demo()
	-- load the image
	I = image.lena():cuda()
	myconverter = convert()
	window = nil
	-- demo the jittering
	for i=1,1000 do
		local O = myconverter(I)--:float()
		if (not window) then
    	window = qtwidget.newwindow(O:size(3), O:size(2))
  	end
		O = image.toDisplayTensor(O)
		displayframe = qtwidget.newimage(O)
		window.port:image(0, 0, displayframe)
		if (0 == i % 100) then
    	collectgarbage()
    	collectgarbage()
  	end
		-- slow down for display
		sleep(1 / 2.0)
	end
end

demo()
