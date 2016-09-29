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
static int hzproc_(Main_affine_fast)(lua_State *L)
/*
 * mapping the image pixels based on the affine transformation matrix
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 2)
    luaL_error(L,  "HZPROC: AffineMap: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* matrix_ = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	/* Check input */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3 ||
			THCTensor_(nDimension)(state, matrix_) != 2)
		luaL_error(L, "HZPROC: AffineMap: incorrect input dims. \n");
	if (THCTensor_(size)(state, matrix_, 0) != 3 ||
			THCTensor_(size)(state, matrix_, 1) != 3 )
		luaL_error(L, "HZPROC: AffineMap: incorrect affine matrix size. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, out_, in_);
	HZAffineFast(state, in_, out_, matrix_);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

static int hzproc_(Main_affine_bili)(lua_State *L)
/*
 * mapping the image pixels based on the affine transformation matrix
 * inputs and output are cuda tensors
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 2)
    luaL_error(L,  "HZPROC: AffineMap: Incorrect number of arguments.\n");
	THCTensor* in_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* matrix_ = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	/* Check input */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, in_) != 3 ||
			THCTensor_(nDimension)(state, matrix_) != 2)
		luaL_error(L, "HZPROC: AffineMap: incorrect input dims. \n");
	if (THCTensor_(size)(state, matrix_, 0) != 3 ||
			THCTensor_(size)(state, matrix_, 1) != 3 )
		luaL_error(L, "HZPROC: AffineMap: incorrect affine matrix size. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, out_, in_);
	HZAffineBili(state, in_, out_, matrix_);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}
