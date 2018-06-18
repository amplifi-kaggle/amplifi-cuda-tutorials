/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

// use this kernel function as-is (no need to change this kernel)
__global__ void kernel(float *a, int offset)
{
	int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
	float x = (float)i;
	float s = sinf(x); 
	float c = cosf(x);
	for(unsigned x=0; x<50; x++) {
	  a[i] = a[i]*a[i] + sqrtf(s*s+c*c+x) + (s-1+x)*(c+1+x)/sqrtf(s*c+x) + x;
	}
}


int main(int argc, char **argv)
{
	// assumption #1: number of threads per CTA (thread-block) is 256
	// assumption #2: number of streams to overlap computation and data transfer = 4
	const int blockSize = 256, nStreams = 4;
	const int n = 4 * 1024 * blockSize * nStreams;
	const int bytes = n * sizeof(float);
	// # of elements to process within each stream
	const int streamSize = n / nStreams;
	const int streamBytes = streamSize * sizeof(float);
   
	// set GPU device
	int devId = 0;
	if (argc > 1) devId = atoi(argv[1]);
	cudaDeviceProp prop;
	checkCuda( cudaGetDeviceProperties(&prop, devId));
	printf("Device : %s\n", prop.name);
	checkCuda( cudaSetDevice(devId) );
  
	// TODO: allocate pinned host memory and device memory (note: we only need 'one' vector array for the purpose of this lab)
	float *a, *d_a;
	// (FILL IN CODE HERE)
	// host memory
	a = (float *) malloc(bytes);

	// Pinned host memory
	cudaHostRegister(&a, bytes, 0);

	// Device mmeory
	cudaMalloc((void **) &d_a, bytes);

	// initialize array  
  memset(a, 0, bytes);

	// create events for latency measurements
	float ms; // elapsed time in milliseconds
	cudaEvent_t startEvent, stopEvent;
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	// TODO: create CUDA streams
	// (FILL IN CODE HERE)
	cudaStream_t stream[nStreams];
	for(int i = 0; i < nStreams; i++) {
		checkCuda( cudaStreamCreate(&stream[i]) );
	}

  // Start timing ...
  checkCuda( cudaEventRecord(startEvent,0) );

	// TODO: baseline case - sequential transfer and execute
	// (FILL IN CODE HERE)

	cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

	kernel<<<streamSize / blockSize, blockSize * nStreams>>>(d_a, 0);

	cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost);

  // stop timer
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);

  // asynchronous version
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

	// TODO: asynchronous version
	// (FILL IN CODE HERE)

	for(int i = 0; i < nStreams; i++)
		cudaMemcpyAsync(d_a, a + streamSize * i, streamBytes, cudaMemcpyHostToDevice, stream[i]);
		// asynchronous kernel invocation
	for(int i = 0; i < nStreams; i++)
		kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, streamSize * i);

	for(int i = 0; i < nStreams; i++)
	 cudaMemcpyAsync(a + streamSize * i, d_a, streamBytes, cudaMemcpyDeviceToHost, stream[i]);

	// stop timer
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
	// measure latency
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );

	// TODO: destroy CUDA streams
	// (FILL IN CODE HERE)
	for(int i = 0; i < nStreams; i++)
	 cudaStreamDestroy(stream[i]);


	// TODO: free CUDA memory
	// (FILL IN CODE HERE)
	cudaFreeHost(a);
	cudaFree(d_a);

  return 0;
}
