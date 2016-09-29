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
#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/hzproc.c"
#else

/* load the implementation detail */
#include "generic/hztable.c"
#include "generic/hzremap.c"
#include "generic/hzaffine.c"
#include "generic/hzcombine.c"
#include "generic/hzcrop.c"
#include "generic/hzflip.c"

// TODO support bicubic mapping
// TODO support radial distortion augmentation

/* register the functions */
static const struct luaL_Reg hzproc_(Remap) [] = 
{
	{"Fast",     hzproc_(Main_map_fast)},
	{"Bilinear", hzproc_(Main_map_bili)},
	{"Combine",  hzproc_(Main_combinetable)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg hzproc_(Transform) [] = 
{
	{"Fast",     hzproc_(Main_affine_fast)},
	{"Bilinear", hzproc_(Main_affine_bili)},
	{"ToTable",  hzproc_(Main_affine_2table)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg hzproc_(Table) [] = 
{
	{"Resize",   hzproc_(Main_table_resize)},
	{"Pad",      hzproc_(Main_table_pad)},
  {"Crop",     hzproc_(Main_table_crop)},
	{"Flip",     hzproc_(Main_table_flip)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg hzproc_(Crop) [] = 
{
	{"Fast",     hzproc_(Main_crop_fast)},
	{"Bilinear", hzproc_(Main_crop_bili)},
	{"Pad",      hzproc_(Main_crop_pad)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg hzproc_(Flip) [] = 
{
	{"Horizon",     hzproc_(Main_flip_horizon)},
	/* end */
	{NULL, NULL}
};


DLL_EXPORT int luaopen_libhzproc(lua_State *L) {
	lua_newtable(L);
	lua_pushvalue(L, -1);
	lua_setglobal(L, "hzproc");
	
	lua_newtable(L);
	luaT_setfuncs(L, hzproc_(Remap), 0);
	lua_setfield(L, -2, "Remap");
	
	lua_newtable(L);
	luaT_setfuncs(L, hzproc_(Transform), 0);
	lua_setfield(L, -2, "Transform");

	lua_newtable(L);
	luaT_setfuncs(L, hzproc_(Table), 0);
	lua_setfield(L, -2, "Table");

	lua_newtable(L);
	luaT_setfuncs(L, hzproc_(Crop), 0);
	lua_setfield(L, -2, "Crop");

	lua_newtable(L);
	luaT_setfuncs(L, hzproc_(Flip), 0);
	lua_setfield(L, -2, "Flip");

	return 1;
}

#endif // THC_GENERIC_FILE
