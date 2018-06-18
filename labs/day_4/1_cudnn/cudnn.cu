#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <iostream>
#include <vector>

#include <cudnn.h>
#include <cublas_v2.h>

#include <unistd.h>
#include <time.h>
#include <pthread.h>


#ifdef USE_CPP_11
#include <thread>
#endif

#define ASSERT_EQ(A, B) {  \
  if((A)!=(B)) { printf("\n\n[CNMEM FAILED]\n"); this->printCnmemMemoryUsage(); assert(0); }        \
}

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}


#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCUBLAS(status) {                                          \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      _error << "CUBLAS failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      assert(0);                                                        \
      FatalError(_error.str());                                        \
    }                                                                  \
}


inline
cudaError_t checkCuda(cudaError_t result)
{

  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }

  return result;
}

#define value_type float
#define DATA_PRECISION  CUDNN_DATA_FLOAT


/*
typedef enum
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6
} cudnnConvolutionFwdAlgo_t;
*/
int main(int argc, char **argv)
{
	// first cmdline parameter is used to specify which CNN algorithm to use for conv derivation
	cudnnConvolutionFwdAlgo_t fwdAlgo	= (cudnnConvolutionFwdAlgo_t)atoi(argv[1]);
	
	// TODO: create streams
	cudaStream_t stream[4];

	for(int i = 0; i < 4; i++) 
		cudaStreamCreate(&stream[i]);
	// ??? 
	// TODO create cudnn handle (search "cudnnCreate()")
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);
  // TODO: interface cudnnHandle with this stream (search "cudnnSetStream()")
	for(int i = 0; i < 4; i++) 
		cudnnSetStream(cudnnHandle, stream[i]);

	//---------------
	// Step #1. input fmap
	//---------------
	// NCHW spec for input feature map (fmap) 
	int n_in	= 64;
	int c_in	= 64;
	int h_in	= 224;
	int w_in	= 224;
	// TODO: create & setup a tensor descriptor (search "cudnnTensorDescriptor_t", "cudnnCreateTensorDescriptor()", and "cudnnSetTensor4dDescriptor()") to keep track of input fmap information

	cudnnTensorDescriptor_t pInputDesc;
	cudnnCreateTensorDescriptor(&pInputDesc);
	cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 
		n_in, c_in, h_in, w_in);

	float *pInput;

	// TODO: malloc GPU memory that can point to input fmap's data elements
	cudaMalloc((void **) &pInput, n_in * c_in * h_in * w_in * sizeof(value_type));

	//---------------
	// Step #2-a. filters
	//---------------
	// KCRS spec for convolutional layer filters
	int k			= 128;
	int r			= 3;
	int s			= 3;
	// TODO: create & setup a filter descriptor (search "cudnnFilterDescriptor_t", "cudnnCreateFilterDescriptor()", and "cudnnSetFilter4dDescriptor()") to keep track of convolutional layer filter information
	cudnnFilterDescriptor_t pFilterDesc;
	cudnnCreateFilterDescriptor(&pFilterDesc);
	cudnnSetFilter4dDescriptor(pFilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,  k, c_in, r, s);
	
	float *pFilter;

	// TODO: malloc GPU memory that can point to filter's data elements
  cudaMalloc((void **) &pFilter, k * c_in * r * s * sizeof(value_type));

	//---------------
	// Step #2-b. Conv layer spec
	//---------------
	// Spec for convolutional layer's operation
	int pad_h	= 1;
	int pad_w	= 1;
	int stride_h	= 1;
	int stride_w	= 1;
	// TODO: create & setup a convolutional layer descriptor (search "cudnnConvolutionDescriptor_t", "cudnnCreateConvolutionDescriptor()", and "cudnnSetConvolution2dDescriptor()") to keep track of information regarding our convolutional layer's pad & stride info
	cudnnConvolutionDescriptor_t pConvDesc;
	cudnnCreateConvolutionDescriptor(&pConvDesc);
	cudnnSetConvolution2dDescriptor(pConvDesc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CONVOLUTION);

	//---------------
	// Step #3. output fmap
	//---------------
	// NCHW spec for output feature map (fmap)
 	// Note) you need to call the right cuDNN call to derive n/c/h/w_out info yourself
	int n_out	= 0;
	int c_out	= 0;
	int h_out	= 0;
	int w_out	= 0;	
	// TODO: find dimension of convolution output (search 'cudnnGetConvolution2dForwardOutputDim()')
	cudnnGetConvolution2dForwardOutputDim(pConvDesc, pInputDesc, pFilterDesc, &n_out, &c_out, &h_out, &w_out);
	// TODO: create & setup a tensor descriptor (search "cudnnTensorDescriptor_t") to keep track of output fmap information
	cudnnTensorDescriptor_t pOutputDesc;
	cudnnCreateTensorDescriptor(&pOutputDesc);
	cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_out, c_out, h_out, w_out);

	float *pOutput;
	// TODO: malloc GPU memory that can point to input fmap's data elements
	cudaMalloc((void **) &pOutput, n_out * c_out * h_out * w_out * sizeof(value_type));

	// allocate workspace if required
	std::cout<<"\n-----------------------\n1. fmap and filter size\n-----------------------"<<std::endl;
	printf("- Input  Fmap size  (N:%4d, C:%4d, H:%4d, W:%4d): ",n_in,c_in,h_in,w_in);
	std::cout<<(n_in*c_in*h_in*w_in*sizeof(value_type))<<" (bytes)"<<std::endl;
	printf("- Filter size       (K:%4d, C:%4d, R:%4d, S:%4d): ",k,c_in,r,s);
	std::cout<<k*c_in*r*s*sizeof(value_type)<<" (bytes)"<<std::endl;
	printf("- Output Fmap size  (N:%4d, C:%4d, H:%4d, W:%4d): ",n_out,c_out,h_out,w_out);
	std::cout<<(n_out*c_out*h_out*w_out*sizeof(value_type))<<" (bytes)\n"<<std::endl;


	//---------------
	// Step #4. Profile all conv algorithms available within cuDNN
	//---------------
	// test which cudnn algorithm 
  int requestedAlgoCount = 6; 
  int returnedAlgoCount;
	// declare fwd-prop performance evaluating data structure profile results (search 'cudnnConvolutionFwdAlgoPerf_t') 
  cudnnConvolutionFwdAlgoPerf_t*        fwdProfileResults;
  fwdProfileResults = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t)*requestedAlgoCount);
  cudnnConvolutionFwdAlgoPerf_t *results = fwdProfileResults;
 
	std::cout<<"\n-----------------------\n2. Profile all algorithm\n-----------------------"<<std::endl;
	// TODO: profile all available (forward propagation) convolutional layer algorithms defined within cuDNN
	// : search 'cudnnFindConvolutionForwardAlgorithm'	
	cudnnFindConvolutionForwardAlgorithm(cudnnHandle, pInputDesc, pFilterDesc, pConvDesc, pOutputDesc, requestedAlgoCount, &returnedAlgoCount, results);
	// TODO: printf (or std::cout) the time taken per each convolutional algorithm execution and the additional workspace required to execute that algorithm
	// algo 0: implicit gemm? (memory 필요없음) / algo 6: winobrad?
	for(int i = 0; i < returnedAlgoCount; i++)
		printf("Algorithm[%u]: Execution time: %f / Workspace taken (bytes): %u\n", results[i].algo, results[i].time, results[i].memory);

  return 0;
}
