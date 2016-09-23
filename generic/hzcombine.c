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
static int hzproc_(Main_affine_2table)(lua_State *L)
/*
 * convert affine transformation matrix to lookup table
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 3)
    luaL_error(L,  "HZPROC: ToTable: Incorrect number of arguments.\n");
	THCTensor* matrix_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
  long width  	= luaL_checknumber(L, 2);
  long height 	= luaL_checknumber(L, 3);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, matrix_) != 2)
		luaL_error(L, "HZPROC: ToTable: incorrect input dims. \n");
	/* Creating coordinate grid */
	long i, j, oidx;
	long chsz = height * width;
	long size = 2 * chsz;
	real *tdata = (real*)THAlloc(sizeof(real)*size);
	/* Init as a grid */
	for (j=0; j<height; j++) {
		for(i=0; i<width; i++) {
			oidx = j*width + i;
			// x channel
			*(tdata + oidx) 			 =  i;
			// y channel
			*(tdata + chsz + oidx) =  j;
		}
	}
	/* Create the CudaTensor */
	THCTensor *in_  = THCTensor_(newWithSize3d)(state, 2, height, 
																width);
	real *ddata = THCTensor_(data)(state, in_);
	THCudaCheck(cudaMemcpy(ddata, tdata, size * sizeof(real), 
							cudaMemcpyHostToDevice));
	THCTensor *out_ = THCTensor_(newWithSize3d)(state, 2, height, 
																width);
	/* Affine transform to grid */
	HZAffineBili(state, in_, out_, matrix_);
	/* Free the memory and tensor */
	THFree(tdata); 
	THCTensor_(free)(state, in_);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

static int hzproc_(Main_combinetable)(lua_State *L)
/*
 * combine lookup two tables
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 2)
    luaL_error(L,  "HZPROC: Combine: Incorrect number of arguments.\n");
	THCTensor* map1_  = *(THCTensor**)luaL_checkudata(L, 1, 
												THC_Tensor);
	THCTensor* map2_ = *(THCTensor**)luaL_checkudata(L, 2, 
												THC_Tensor);
	/* Check input dims */
	THCState *state = cutorch_getstate(L);
	if (THCTensor_(nDimension)(state, map1_) != 3 ||
			THCTensor_(nDimension)(state, map2_) != 3)
		luaL_error(L, "HZPROC: Combine: incorrect input dims. \n");
	/* Init output tensor */
	THCTensor* out_ =  THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, out_, map2_);
	HZMapBili(state, map1_, out_, map2_);
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
  luaT_pushudata(L, (void*)out_, THC_Tensor);
	/* C function return number of the outputs */
	return 1;
}

