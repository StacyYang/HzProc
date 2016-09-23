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
#include "generic/hzlookuptable.c"
#include "generic/hzremap.c"
#include "generic/hzcombine.c"

// TODO support color and lighting jittering
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
	{"Resize",   hzproc_(Main_resizetable)},
	{"Pad",      hzproc_(Main_padtable)},
  {"Crop",     hzproc_(Main_croptable)},
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
	return 1;
}

#endif // THC_GENERIC_FILE
