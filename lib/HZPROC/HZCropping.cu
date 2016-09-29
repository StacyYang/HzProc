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
__global__ void HZCrop_Fast_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output,
	int xb, int yb, real sx, real sy)
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
	xi = xb + sx*xo;
	yi = yb + sy*yo;
	/* boundary check for input*/
	if(xi >= 0 && xi < inwidth && yi >=0 && yi < inheight)
		output[ch][yo][xo] = input[ch][yi][xi].ldg();
	else
		output[ch][yo][xo] = 0;
}

void HZCropFast(THCState *state, THCTensor *input_, THCTensor *output_,
							int xi, int yi, int xo, int yo)
/*
 * mapping the image pixels based on the lookuptable
 */
{
  /* declarations of the variables */
	int inw, inh, ow, oh;
	real sx, sy;
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 2, input_, output_);
	/* Device tensors */
	THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
	/* scale params */
	ow = input.getSize(2);
	oh = input.getSize(1);
	inw = xo - xi;
	inh = yo - yi;
	sx = 1.0 * ow / inw;
	sy = 1.0 * oh / inh;
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	HZCrop_Fast_kernel<<<blocks, threads, 0, stream>>>(input, output, 
										xi, yi, sx, sy);
	THCudaCheck(cudaGetLastError());
}

__global__ void HZCrop_Bili_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output,
	int xb, int yb, real sx, real sy)
{
  /* declarations of the variables */
  int ch, xo, yo, inwidth, inheight;
	real xi, yi, wx, wy, w00, w01, w10, w11;
	int x0, y0;
  /* Get the index and channels */ 
  ch = blockIdx.z;
  xo = blockIdx.x * blockDim.x + threadIdx.x;
  yo = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (xo >= output.getSize(2) || yo >= output.getSize(1))	return;
	inwidth  = input.getSize(2);
	inheight = input.getSize(1);
	/* main operation */
	xi = xb + sx*xo;
	yi = yb + sy*yo;
	x0 = (int)xi;
	y0 = (int)yi;
	/* boundary check for input*/
	if(xi >= 0 && xi < inwidth && yi >=0 && yi < inheight)
	{
		// TODO FIXME there are bugs in crop bilinear kernel function
		wx = 0.5;//1.0 - (xi - x0);
		wy = 0.5;//1.0 - (yi - y0);
		w00 = wx * wy;
		w01 = (1-wx) * wy;
		w10 = wx * (1-wy);
		w11 = (1-wx) * (1-wy);
		output[ch][yo][xo] =  w00*input[ch][y0  ][x0  ].ldg()
												+ w01*input[ch][y0  ][x0+1].ldg()
												+ w10*input[ch][y0+1][x0  ].ldg()
												+ w11*input[ch][y0+1][x0+1].ldg();
	}
	else
		output[ch][yo][xo] = 0;
}

void HZCropBili(THCState *state, THCTensor *input_, THCTensor *output_,
							int xi, int yi, int xo, int yo)
/*
 * mapping the image pixels based on the lookuptable
 */
{
  /* declarations of the variables */
	int inw, inh, ow, oh;
	real sx, sy;
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 2, input_, output_);
	/* Device tensors */
	THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
	/* scale params */
	ow = input.getSize(2);
	oh = input.getSize(1);
	inw = xo - xi;
	inh = yo - yi;
	sx = 1.0 * ow / inw;
	sy = 1.0 * oh / inh;
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	HZCrop_Bili_kernel<<<blocks, threads, 0, stream>>>(input, output, 
										xi, yi, sx, sy);
	THCudaCheck(cudaGetLastError());
}

__global__ void HZCrop_Pad_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output,
	int xb, int yb, int pad)
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
	xi = xo + xb - pad;
	yi = yo + yb - pad;
	/* boundary check for input*/
	if(xi >= 0 && xi < inwidth && yi >=0 && yi < inheight)
		output[ch][yo][xo] = input[ch][yi][xi].ldg();
	else
		output[ch][yo][xo] = 0;
}

void HZCropPad(THCState *state, THCTensor *input_, THCTensor *output_,
							int xi, int yi, int pad)
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
	HZCrop_Pad_kernel<<<blocks, threads, 0, stream>>>(input, output, 
										xi, yi, pad);
	THCudaCheck(cudaGetLastError());
}

