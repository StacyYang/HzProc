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
#include "TH.h"
#include "luaT.h"
#include <THC/THC.h>
#include "lib/HZPROC/HZPROC.h"

/* extern function in cutorch */
struct THCState;
#ifdef __cplusplus
extern "C" struct THCState* cutorch_getstate(lua_State* L);
#else
extern struct THCState* cutorch_getstate(lua_State* L);
#endif

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define hzproc_(NAME) TH_CONCAT_3(hzproc_, Real, NAME)

#ifdef __cplusplus
extern "C" {
#endif

#include "generic/hzproc.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libhzproc(lua_State *L) {
	lua_newtable(L);
	lua_pushvalue(L, -1);
	lua_setglobal(L, "hzproc");
	
	lua_newtable(L);
	luaT_setfuncs(L, hzproc_FloatRemap, 0);
	lua_setfield(L, -2, "Remap");
	
	lua_newtable(L);
	luaT_setfuncs(L, hzproc_FloatTable, 0);
	lua_setfield(L, -2, "Table");
	return 1;
}

#ifdef __cplusplus
}
#endif
