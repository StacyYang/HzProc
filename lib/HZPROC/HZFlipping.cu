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
__global__ void HZFlip_Horizon_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output)
{
  /* declarations of the variables */
  int ch, xo, yo, xi, yi, inwidth, inheight;
  /* Get the index and channels */ 
  ch = blockIdx.z;
  xo = blockIdx.x * blockDim.x + threadIdx.x;
  yo = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (xo >= output.getSize(2) || yo >= output.getSize(1))	return;
	inwidth  = input.getSize(2);
	inheight = input.getSize(1);
	/* main operation */
	xi = inwidth  - 1 - xo;
	yi = yo;
	/* boundary check for input*/
	if(xi >= 0 && xi < inwidth && yi >=0 && yi < inheight)
		output[ch][yo][xo] = input[ch][yi][xi].ldg();
	else
		output[ch][yo][xo] = 0;
}

void HZFlipHorizon(THCState *state, THCTensor *input_, THCTensor *output_)
/*
 * mapping the image pixels based on the lookuptable
 */
{
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 2, input_, output_);
	/* Device tensors */
	THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	HZFlip_Horizon_kernel<<<blocks, threads, 0, stream>>>(input, output); 
	THCudaCheck(cudaGetLastError());
}

