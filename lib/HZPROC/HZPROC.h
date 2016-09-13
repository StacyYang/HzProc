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
#ifndef __HZPROC_H__
#define __HZPROC_H__

#include <THC/THC.h>
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* HZMapping.cu */
void HZMapping(THCState *state, THCudaTensor *input, THCudaTensor *output,
							THCudaTensor *table);
/* HZAffineMap.cu */
void HZAffineMap(THCState *state, THCudaTensor *input, THCudaTensor *output,
							THCudaTensor *matrix);

#ifdef __cplusplus
}
#endif

#endif //__HZPROC_H__
