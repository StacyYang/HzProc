/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 * Created by: Hang Zhang
 * ECE Department, Rutgers University
 * Email: zhang.hang@rutgers.edu
 * Copyright (c) 2016
 *
 * Feel free to reuse and distribute this software for research or 
 * non-profit purpose, subject to the following conditions:
 *  1. The code must retain the above copyright notice, this list of
 *     conditions.
 *  2. Original authors' names are not deleted.
 *  3. The authors' names are not used to endorse or promote products
 *      derived from this software 
 *+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 */
static int hzproc_(Main_crop_fast)(lua_State *L)
/*
 * crop the image 
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 7)
    luaL_error(L,  "HZPROC: Crop: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
  long width 	= luaL_checknumber(L, 2);
  long height = luaL_checknumber(L, 3);
  long xi = luaL_checknumber(L, 4);
  long yi = luaL_checknumber(L, 5);
  long xo = luaL_checknumber(L, 6);
  long yo = luaL_checknumber(L, 7);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3)
		luaL_error(L, "HZPROC: Crop: incorrect input dims. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(newWithSize3d)(state, 
														THCTensor_(size)(state, in_, 0),
														width, height);
	HZCropFast(state, in_, out_, xi, yi, xo, yo);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

static int hzproc_(Main_crop_bili)(lua_State *L)
/*
 * crop the image 
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 7)
    luaL_error(L,  "HZPROC: Crop: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
  long width 	= luaL_checknumber(L, 2);
  long height = luaL_checknumber(L, 3);
  long xi = luaL_checknumber(L, 4);
  long yi = luaL_checknumber(L, 5);
  long xo = luaL_checknumber(L, 6);
  long yo = luaL_checknumber(L, 7);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3)
		luaL_error(L, "HZPROC: Crop: incorrect input dims. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(newWithSize3d)(state, 
														THCTensor_(size)(state, in_, 0),
														width, height);
	HZCropBili(state, in_, out_, xi, yi, xo, yo);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

static int hzproc_(Main_crop_pad)(lua_State *L)
/*
 * crop the image 
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 6)
    luaL_error(L,  "HZPROC: Crop: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
  long xi = luaL_checknumber(L, 2);
  long yi = luaL_checknumber(L, 3);
  long width 	= luaL_checknumber(L, 4);
  long height = luaL_checknumber(L, 5);
  long pad    = luaL_checknumber(L, 6);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3)
		luaL_error(L, "HZPROC: Crop: incorrect input dims. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(newWithSize3d)(state, 
														THCTensor_(size)(state, in_, 0),
														width, height);
	HZCropPad(state, in_, out_, xi, yi, pad);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}
