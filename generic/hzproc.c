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

/* hzlookuptable.c */
static int hzproc_(Main_resizetable)(lua_State *L);
static int hzproc_(Main_padtable)(lua_State *L);
static int hzproc_(Main_croptable)(lua_State *L);
/* hzremap.c TODO biliear mapping */
static int hzproc_(Main_mapping)(lua_State *L);
static int hzproc_(Main_affinemapping)(lua_State *L);
// TODO 
static int hzproc_(Main_combinetable)(lua_State *L);
// TODO support Radial distortion augmentation

/* register the functions */
static const struct luaL_Reg hzproc_(Remap) [] = 
{
	/* hzremap.c */
	{"Fast", hzproc_(Main_mapping)},
	{"Affine", hzproc_(Main_affinemapping)},
	/* end */
	{NULL, NULL}
};

static const struct luaL_Reg hzproc_(Table) [] = 
{
	/* hzlookuptable.c */
	{"Resize", hzproc_(Main_resizetable)},
	{"Pad", hzproc_(Main_padtable)},
  {"Crop", hzproc_(Main_croptable)},
	/* end */
	{NULL, NULL}
};
/* load the implementation detail */
#include "generic/hzlookuptable.c"
#include "generic/hzremap.c"

#endif // THC_GENERIC_FILE
