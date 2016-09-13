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
#include "HZPROC.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void HZMapping_kernel (
	THCDeviceTensor<float, 3> input,
	THCDeviceTensor<float, 3> output,
	THCDeviceTensor<float, 3> table)
{
  /* declarations of the variables */
  int ch, xo, yo, xi, yi, inwidth, inheight;
  /* Get the index and channels */ 
  ch = blockIdx.z;
  xo = blockIdx.x * blockDim.x + threadIdx.x;
  yo = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (xo >= output.getSize(2), yo >= output.getSize(1))	return;
	inwidth  = input.getSize(2);
	inheight = input.getSize(1);
	/* main operation */
	xi = table[0][yo][xo];
	yi = table[1][yo][xo];
	if(xi >= 0 && xi < inwidth && yi >-0 && yi < inheight)
		output[ch][yo][xo] = input[ch][yi][xi].ldg();
	else
		output[ch][yo][xo] = 0;
}

void HZMapping(THCState *state, THCudaTensor *input_, THCudaTensor *output_,
							THCudaTensor *table_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 3, input_, output_, table_);
	/* Device tensors */
	THCDeviceTensor<float, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<float, 3> output = devicetensor<3>(state, output_);
	THCDeviceTensor<float, 3> table  = devicetensor<3>(state, table_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	
	HZMapping_kernel<<<blocks, threads, 0, stream>>>(input, output, table);
	THCudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
