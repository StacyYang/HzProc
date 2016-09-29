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
static int hzproc_(Main_flip_horizon)(lua_State *L)
/*
 * flip the image pixels on horizontal
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 1)
    luaL_error(L,  "HZPROC: Flip: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3)
		luaL_error(L, "HZPROC: Flip: incorrect input dims. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, out_, in_);
	HZFlipHorizon(state, in_, out_);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

