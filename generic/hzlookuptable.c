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
static int hzproc_(Main_resizetable)(lua_State *L)
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
	/* continous ptr */
	real *tdata;
  THTensor *tensor 	= THTensor_(newWithSize3d)(2, out_height, out_width);
  tdata = THTensor_(data)(tensor);
	/* Creating lookup table */
	long i, j, oidx;
	long chsz = out_height * out_width;
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
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, torch_Tensor);
	/* C function return number of the outputs */
  return 1;
}

static int hzproc_(Main_padtable)(lua_State *L)
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
	real *tdata;
  THTensor *tensor 	= THTensor_(newWithSize3d)(2, out_height, out_width);
  tdata = THTensor_(data)(tensor);
	/* Creating lookup table */
	long i, j, oidx, cxin, cyin, cxo, cyo;
	long chsz = out_height * out_width;
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
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, torch_Tensor);
	/* C function return number of the outputs */
  return 1;
}

static int hzproc_(Main_croptable)(lua_State *L)
/*
 * Generating lookup table for croping
 */
{
	/* Check number of inputs */
  if(lua_gettop(L) != 6)
    luaL_error(L, "HZPROC: GetCropTabel: Incorrect number of arguments.\n");
	/* Get input variables */
  long in_width 	= luaL_checknumber(L, 1);
  long in_height 	= luaL_checknumber(L, 2);
  long out_width  = luaL_checknumber(L, 3);
  long out_height = luaL_checknumber(L, 4);
  long x_offset   = luaL_checknumber(L, 5);
  long y_offset   = luaL_checknumber(L, 6);
	/* continous ptr */
	real *tdata;
  THTensor *tensor 	= THTensor_(newWithSize3d)(2, out_height, out_width);
  tdata = THTensor_(data)(tensor);
	/* Creating lookup table */
	long i, j, oidx;
	long chsz = out_height * out_width;
	/* centering and recentering */
	for (j=0; j<out_height; j++) {
		for(i=0; i<out_width; i++) {
			oidx = j*out_width + i;
			*(tdata + oidx)        = i + x_offset;
			*(tdata + chsz + oidx) = j + y_offset;
		}
	}
	/* return the tensor */
	lua_pop(L, lua_gettop(L));
	luaT_pushudata(L, (void*)tensor, torch_Tensor);
	/* C function return number of the outputs */
  return 1;
}
