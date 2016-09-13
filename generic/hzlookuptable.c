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
static int hzproc_(Main_scaletable)(lua_State *L)
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


