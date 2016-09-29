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
static int hzproc_(Main_table_resize)(lua_State *L)
/*
 * Generating lookup table to scale the 
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 4)
    luaL_error(L, "HZPROC: GetResizeTabel: Incorrect number of arguments.\n");
	/* Get input variables */
  long in_width 	= luaL_checknumber(L, 1);
  long in_height 	= luaL_checknumber(L, 2);
  long out_width  = luaL_checknumber(L, 3);
  long out_height = luaL_checknumber(L, 4);
	/* Creating lookup table */
	long i, j, oidx;
	long chsz = out_height * out_width;
	long size = 2 * chsz;
	real *tdata = (real*)THAlloc(sizeof(real)*size);
	real scx = 1.0 * in_width  / out_width;
	real scy = 1.0 * in_height / out_height;

	for (j=0; j<out_height; j++) {
		for(i=0; i<out_width; i++) {
			oidx = j*out_width + i;
			// x channel
			*(tdata + oidx) 			 = scx * i;
			// y channel
			*(tdata + chsz + oidx) = scy * j;
		}
	}
	/* Create the CudaTensor and copy the data */
	THCState *state = cutorch_getstate(L);
	THCTensor *tensor = THCTensor_(newWithSize3d)(state, 2, out_height, 
																out_width);
	real *ddata = THCTensor_(data)(state, tensor);
	THCudaCheck(cudaMemcpy(ddata, tdata, size * sizeof(real), 
							cudaMemcpyHostToDevice));
	THFree(tdata); 
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, THC_Tensor);
	/* C function return number of the outputs */
  return 1;
}

static int hzproc_(Main_table_pad)(lua_State *L)
/*
 * Generating lookup table for padding
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 4)
    luaL_error(L, "HZPROC: GetPadTabel: Incorrect number of arguments.\n");
	/* Get input variables */
  long in_width 	= luaL_checknumber(L, 1);
  long in_height 	= luaL_checknumber(L, 2);
  long out_width  = luaL_checknumber(L, 3);
  long out_height = luaL_checknumber(L, 4);
	/* continous ptr */
	long chsz = out_height * out_width;
	long size = 2*chsz;
	real *tdata = (real*)THAlloc(sizeof(real)*size);
	/* Creating lookup table */
	long i, j, oidx, cxin, cyin, cxo, cyo;
	cxin = (in_width   - 1) / 2; 
	cyin = (in_height  - 1) / 2; 
	cxo  = (out_width  - 1) / 2; 
	cyo  = (out_height - 1) / 2; 
	/* centering and recentering */
	for (j=0; j<out_height; j++) {
		for(i=0; i<out_width; i++) {
			oidx = j*out_width + i;
			*(tdata + oidx) = i-cxo + cxin;
			*(tdata + chsz + oidx) = j-cyo + cyin;
		}
	}
	/* Create the CudaTensor and copy the data */
	THCState *state = cutorch_getstate(L);
	THCTensor *tensor = THCTensor_(newWithSize3d)(state, 2, out_height, 
																out_width);
	real *ddata = THCTensor_(data)(state, tensor);
	THCudaCheck(cudaMemcpy(ddata, tdata, size * sizeof(real), 
							cudaMemcpyHostToDevice));
	THFree(tdata); 
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, THC_Tensor);
	/* C function return number of the outputs */
  return 1;
}

static int hzproc_(Main_table_crop)(lua_State *L)
/*
 * Generating lookup table for croping
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 6)
    luaL_error(L, "HZPROC: Table.Crop: Incorrect number of arguments.\n");
	/* Get input variables */
  long in_width 	= luaL_checknumber(L, 1);
  long in_height 	= luaL_checknumber(L, 2);
  long out_width  = luaL_checknumber(L, 3);
  long out_height = luaL_checknumber(L, 4);
  long x_offset   = luaL_checknumber(L, 5);
  long y_offset   = luaL_checknumber(L, 6);
	/* continous ptr */
	long chsz = out_height * out_width;
	long size = 2*chsz;
	real *tdata = (real*)THAlloc(sizeof(real)*size);
	/* Creating lookup table */
	long i, j, oidx;
	/* centering and recentering */
	for (j=0; j<out_height; j++) {
		for(i=0; i<out_width; i++) {
			oidx = j*out_width + i;
			*(tdata + oidx)        = i + x_offset;
			*(tdata + chsz + oidx) = j + y_offset;
		}
	}
	/* Create the CudaTensor and copy the data */
	THCState *state = cutorch_getstate(L);
	THCTensor *tensor = THCTensor_(newWithSize3d)(state, 2, out_height, 
																out_width);
	real *ddata = THCTensor_(data)(state, tensor);
	THCudaCheck(cudaMemcpy(ddata, tdata, size * sizeof(real), 
							cudaMemcpyHostToDevice));
	THFree(tdata); 
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, THC_Tensor);
	/* C function return number of the outputs */
  return 1;
}

static int hzproc_(Main_table_flip)(lua_State *L)
/*
 * Generating lookup table for horizontal flioping
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 2)
    luaL_error(L, "HZPROC: Table.Flip: Incorrect number of arguments.\n");
	/* Get input variables */
  long width 	= luaL_checknumber(L, 1);
  long height 	= luaL_checknumber(L, 2);
	/* continous ptr */
	long chsz = height*width;
	long size = 2*chsz;
	real *tdata = (real*)THAlloc(sizeof(real)*size);
	/* Creating lookup table */
	long i, j, oidx;
	/* centering and recentering */
	for (j=0; j<height; j++) {
		for(i=0; i<width; i++) {
			oidx = j*width + i;
			*(tdata + oidx)        = (width-1) - i;
			*(tdata + chsz + oidx) = j;
		}
	}
	/* Create the CudaTensor and copy the data */
	THCState *state = cutorch_getstate(L);
	THCTensor *tensor = THCTensor_(newWithSize3d)(state, 2, height, 
																width);
	real *ddata = THCTensor_(data)(state, tensor);
	THCudaCheck(cudaMemcpy(ddata, tdata, size * sizeof(real), 
							cudaMemcpyHostToDevice));
	THFree(tdata); 
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, THC_Tensor);
	/* C function return number of the outputs */
  return 1;
}

