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

local Affine = {}

Affine.Scale = function(sx, sy)
	local t = {}
	t[1] = { sx, 0, 0}
	t[2] = { 0, sy, 0}
	t[3] = { 0, 0,  1}
	return torch.CudaTensor(t)
end

Affine.Shift = function (tx,ty)
	local t = {}
	t[1] = { 1, 0, tx}
	t[2] = { 0, 1, ty}
	t[3] = { 0, 0,  1}
	return torch.CudaTensor(t)
end

Affine.Rotate = function(theta)
	local t = {}
	t[1] = {math.cos(theta),-math.sin(theta), 0}
	t[2] = {math.sin(theta), math.cos(theta), 0}
	t[3] = {0,            0,                  1}
	return torch.CudaTensor(t)
end

Affine.Shear = function (kx,ky)
	local t = setmetatable({},mt)
	t[1] = { 1, kx, 0}
	t[2] = { ky, 1, 0}
	t[3] = { 0, 0,  1}
	return torch.CudaTensor(t)
end

return Affine
