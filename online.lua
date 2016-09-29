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

-- The color augmentation code are copied from fb.resnet.torch
-- original copyright preserves

require 'hzproc'

local M = {}

function M.Compose(transforms)
  return function(input)
    for _, transform in ipairs(transforms) do
      input = transform(input)
    end
    return input
  end
end

function M.ColorNormalize(meanstd)
  return function(img)
    img = img:clone()
    for i=1,3 do
      img[i]:add(-meanstd.mean[i])
      img[i]:div(meanstd.std[i])
    end
    return img
  end
end

function M.Warp(sh, deg, sc, asp)
	return function(img)
		sc  = sc or 2
		asp = asp or 4.0/3
		-- randomly init the params
		-- shearing and rotate
		local shx = (torch.uniform()-0.5)*sh;
		local shy = (torch.uniform()-0.5)*sh;
		local theta  = (torch.uniform() - 0.5) * deg * math.pi / 180
		-- scaling params
		local scale  = torch.uniform(1/sc, sc)
		local aspect = torch.uniform(1/asp, asp)
		local scx = math.sqrt(scale * aspect)
		local scy = math.sqrt(scale / aspect)
		-- get the affine matrix
		local mat = hzproc.Affine.ScaleArround(scx, scy,
																		(img:size(3)+1)/2, 
																		(img:size(2)+1)/2)
		mat = mat * hzproc.Affine.ShearArround(shx, shy,
																		(img:size(3)+1)/2, 
																		(img:size(2)+1)/2)
		mat = mat * hzproc.Affine.RotateArround(theta, 
																		(img:size(3)+1)/2, 
																		(img:size(2)+1)/2)
		-- affine transform
		return hzproc.Transform.Bilinear(img, mat);
	end
end

-- Resize the input image
function M.Resize(inw, inh, ow, oh)
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

-- Crop to centered rectangle
function M.CenterCrop(ow, oh)
  return function(input)
		-- avoid unnecessary opterations
	  local w, h = input:size(3), input:size(2)
    if (w == ow and h == oh) then
      return input
    end	
		local xoff = (w-ow) / 2
		local yoff = (h-oh) / 2
		-- mapping 
		return hzproc.Crop.Fast(input, ow, oh, xoff, yoff, 
																	xoff + ow, yoff + oh)
  end
end

-- Random crop form larger image 
function M.RandomCrop(size, pad)
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

-- Four corner patches and center crop from image and its horizontal 
-- reflection
function M.TenCrop(size)
  local centerCrop = M.CenterCrop(size)
  return function(input)
    local w, h = input:size(3), input:size(2)
    local output = {}
    for _, img in ipairs{input, hzproc.Flip.Horizon(input)} do
      table.insert(output, centerCrop(img))
      table.insert(output, hzproc.Crop.Fast(img, size, size, 
		 																	0, 0,	size, size))
      table.insert(output, hzproc.Crop.Fast(img, size, size, 
		 																	w-size, 0, w, size))
      table.insert(output, hzproc.Crop.Fast(img, size, size, 
		 																	0, h-size, size, h))
      table.insert(output, hzproc.Crop.Fast(img, size, size, 
		 																	w-size, h-size, w, h))
    end

    -- View as mini-batch
    for i, img in ipairs(output) do
      output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
    end
      return input.cat(output, 1)
   end
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 
-- (Inception-style)
function M.RandomSizedCrop(size)
  local crop = M.RandomCrop(size, size)
  return function(input)
    local attempt = 0
    repeat
      local area = input:size(2) * input:size(3)
      local targetArea = torch.uniform(0.08, 1.0) * area

      local aspectRatio = torch.uniform(3/4, 4/3)
      local w = torch.round(math.sqrt(targetArea * aspectRatio))
      local h = torch.round(math.sqrt(targetArea / aspectRatio))

      if torch.uniform() < 0.5 then
        w, h = h, w
      end

      if h <= input:size(2) and w <= input:size(3) then
        local y1 = torch.random(0, input:size(2) - h)
        local x1 = torch.random(0, input:size(3) - w)

        return hzproc.Crop.Fast(input, size, size,
															x1, y1, x1 + w, y1 + h)
      end
      attempt = attempt + 1
    until attempt >= 10
    -- fallback
    return crop(input)
  end
end

function M.HorizontalFlip(prob)
  return function(input)
    if torch.uniform() < prob then
      input = hzproc.Flip.Horizon(input)
    end
    return input
  end
end

-- Lighting noise (AlexNet-style PCA-based noise)
function M.Lighting(alphastd, eigval, eigvec)
  return function(input)
    if alphastd == 0 then
      return input
    end

    local alpha = torch.Tensor(3):normal(0, alphastd)
    local rgb = eigvec:clone()
      :cmul(alpha:view(1, 3):expand(3, 3))
      :cmul(eigval:view(1, 3):expand(3, 3))
      :sum(2)
      :squeeze()

    input = input:clone()
    for i=1,3 do
      input[i]:add(rgb[i])
    end
    return input
  end
end

local function blend(img1, img2, alpha)
  return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
  dst:resizeAs(img)
  dst[1]:zero()
  dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
  dst[2]:copy(dst[1])
  dst[3]:copy(dst[1])
  return dst
end

function M.Saturation(var)
  local gs

  return function(input)
    gs = gs or input.new()
    grayscale(gs, input)

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
  end
end

function M.Brightness(var)
  local gs

  return function(input)
    gs = gs or input.new()
    gs:resizeAs(input):zero()

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
  end
end

function M.Contrast(var)
  local gs

  return function(input)
    gs = gs or input.new()
    grayscale(gs, input)
    gs:fill(gs[1]:mean())

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
  end
end

function M.RandomOrder(ts)
  return function(input)
    local img = input.img or input
    local order = torch.randperm(#ts)
    for i=1,#ts do
      img = ts[order[i]](img)
    end
    return input
  end
end

function M.ColorJitter(opt)
  local brightness = opt.brightness or 0
  local contrast = opt.contrast or 0
  local saturation = opt.saturation or 0

  local ts = {}
  if brightness ~= 0 then
    table.insert(ts, M.Brightness(brightness))
  end
  if contrast ~= 0 then
    table.insert(ts, M.Contrast(contrast))
  end
  if saturation ~= 0 then
    table.insert(ts, M.Saturation(saturation))
  end

  if #ts == 0 then
    return function(input) return input end
  end

  return M.RandomOrder(ts)
end

return M
