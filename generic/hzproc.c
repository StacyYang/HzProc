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
#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/hzproc.c"
#else

/* functions declearations */
// TODO cropping & padding
static int hzproc_(Main_scaletable)(lua_State *L);
static int hzproc_(Main_mapping)(lua_State *L);
// TODO affine to lookuptable
static int hzproc_(Main_affinemapping)(lua_State *L);
// TODO 
static int hzproc_(Main_combinetable)(lua_State *L);

/* register the functions */
static const struct luaL_Reg hzproc_(Main__) [] = 
{
	{"GetResizeTable", hzproc_(Main_scaletable)},
	{"Remap", hzproc_(Main_mapping)},
	{"AffineMap", hzproc_(Main_affinemapping)},
	{NULL, NULL}
};

/* load the implementation detail */
#include "generic/hzlookuptable.c"
#include "generic/hzmap.c"

#endif // TH_GENERIC_FILE
