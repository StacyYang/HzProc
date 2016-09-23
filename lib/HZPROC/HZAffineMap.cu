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
__global__ void HZAffine_Fast_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output,
	THCDeviceTensor<real, 2> matrix)
{
  /* declarations of the variables */
  int ch, xo, yo, xi, yi, inwidth, inheight;
  real xi_, yi_, zi_;
	/* Get the index and channels */ 
  ch = blockIdx.z;
  xo = blockIdx.x * blockDim.x + threadIdx.x;
  yo = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (xo >= output.getSize(2) || yo >= output.getSize(1))	return;
	inwidth  = input.getSize(2);
	inheight = input.getSize(1);
	/* main operation */
	xi_ = matrix[0][0]*xo + matrix[1][0]*yo + matrix[2][0];
	yi_ = matrix[0][1]*xo + matrix[1][1]*yo + matrix[2][1];
	zi_ = matrix[0][2]*xo + matrix[1][2]*yo + matrix[2][2];
	xi = (int) (xi_ / zi_);
	yi = (int) (yi_ / zi_);
	/* boundary check for input*/
	if(xi >= 0 && xi < inwidth && yi >=0 && yi < inheight)
		output[ch][yo][xo] = input[ch][yi][xi].ldg();
	else
		output[ch][yo][xo] = 0;
}

void HZAffineFast(THCState *state, THCTensor *input_, 
							THCTensor *output_, THCTensor *matrix_)
/*
 * mapping the image pixels based on the inversed
 * affine transformation matrix
 */
{
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 3, input_, output_, matrix_);
	/* Device tensors */
	THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
	/* inverse the affine matrix */
	THCTensor *mat_ = THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, mat_, matrix_);
	THCTensor_(getri)(state, mat_, matrix_);
	THCDeviceTensor<real, 2> matrix = devicetensor<2>(state, mat_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	HZAffine_Fast_kernel<<<blocks, threads, 0, stream>>>(input, output, matrix);
	THCudaCheck(cudaGetLastError());
}

__global__ void HZAffine_Bili_kernel (
	THCDeviceTensor<real, 3> input,
	THCDeviceTensor<real, 3> output,
	THCDeviceTensor<real, 2> matrix)
{
  /* declarations of the variables */
  int ch, xo, yo, x0, y0, inwidth, inheight;
	real xi, yi, wx, wy, w00, w01, w10, w11;
	real xi_, yi_, zi_;
	/* Get the index and channels */ 
  ch = blockIdx.z;
  xo = blockIdx.x * blockDim.x + threadIdx.x;
  yo = blockIdx.y * blockDim.y + threadIdx.y;
	/* boundary check for output */
	if (xo >= output.getSize(2) || yo >= output.getSize(1))	return;
	inwidth  = input.getSize(2);
	inheight = input.getSize(1);
	/* main operation */
	xi_ = matrix[0][0]*xo + matrix[1][0]*yo + matrix[2][0];
	yi_ = matrix[0][1]*xo + matrix[1][1]*yo + matrix[2][1];
	zi_ = matrix[0][2]*xo + matrix[1][2]*yo + matrix[2][2];
	xi = xi_ / zi_;
	yi = yi_ / zi_;
	x0 = (int)xi;
	y0 = (int)yi;
	/* boundary check for input*/
	if(x0 >= 0 && x0 < inwidth-1 && y0 >=0 && yi < inheight-1)
	{
		wx = 1.0 - (xi - x0);
		wy = 1.0 - (yi - y0);
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

void HZAffineBili(THCState *state, THCTensor *input_, 
							THCTensor *output_, THCTensor *matrix_)
/*
 * mapping the image pixels based on the inversed
 * affine transformation matrix
 */
{
	/* Check the GPU index */
	HZPROC_assertSameGPU(state, 3, input_, output_, matrix_);
	/* Device tensors */
	THCDeviceTensor<real, 3> input  = devicetensor<3>(state, input_);
	THCDeviceTensor<real, 3> output = devicetensor<3>(state, output_);
	/* inverse the affine matrix */
	THCTensor *mat_ = THCTensor_(new)(state);
	THCTensor_(resizeAs)(state, mat_, matrix_);
	THCTensor_(getri)(state, mat_, matrix_);
	THCDeviceTensor<real, 2> matrix = devicetensor<2>(state, mat_);
	/* kernel function */
	cudaStream_t stream = THCState_getCurrentStream(state);
	dim3 threads(16, 16);
	dim3 blocks(output.getSize(2)/16+1, output.getSize(1)/16+1, 
							output.getSize(0));
	HZAffine_Bili_kernel<<<blocks, threads, 0, stream>>>(input, output, matrix);
	THCudaCheck(cudaGetLastError());
}
