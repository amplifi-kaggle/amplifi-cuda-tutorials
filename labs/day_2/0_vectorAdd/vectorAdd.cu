#include<stdio.h>
#include<cuda_runtime.h>

/**
 * CUDA kernel code
 */
__global__
void vectorAdd(float *A,  float *B, float *C, int numElements)
{
	//TODO: derive the indices of A/B/C[] yourself using 'threadIdx.x, blockDim.x, blockIdx.x'
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//vector addition
	
	//TODO: C = A + B
	if(i < numElements) {
		C[i] = A[i] + B[i];
	}

}

/**
 * Host main routine
 */
int main(void)
{
	cudaError_t err = cudaSuccess;

	int n = 50000;
	size_t size = n * sizeof(float);
	// alloc host side memory
	//TODO: malloc the host memeory
	float *h_A = (float *)malloc(size);
	float *h_B = (float *)malloc(size);
	float *h_C = (float *)malloc(size);

	//alloc device vetors
	float *d_A = NULL;
	float *d_B = NULL;
	float *d_C = NULL;
	//TODO: use cudaMalloc(void**, int) to allocate device memory
	err = cudaMalloc((void **)&d_A, size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));	
	}
	err = cudaMalloc((void **)&d_B, size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void **)&d_C, size);
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
	}
	
	//init vector A and vector B
	///TODO: h_A[i] = random number 0 ~ 1 <use rand()>
	for(int i = 0; i < n; i++) {
		h_A[i] = rand() % 2;
		h_B[i] = rand() % 2;
	}

	// copy host data to device
	printf("Copy input vectors to device\n");
	//TODO: use cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

	//Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
	printf("CUDA kernel launch with %d blocks of %d threads \n", blocksPerGrid, threadsPerBlock);
	//TODO: call vectorAdd function!
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
	err = cudaGetLastError();

	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//Copy device output data to host
	printf("Copy output data to host\n");
	//TODO: cudaMemcpy(dest,source,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	//Verifiy output
	int pass = 0;
	//TODO: Uncommnet
	
	pass = 1;
	for (int i=0;i<n;i++)
	{
		if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			pass = 0;
			fprintf(stderr, "Result is invalid at element %d!\n",i);
			exit(EXIT_FAILURE);
		}
	}
	
	if (pass)
		printf("Test PASSED\n");
	else
		printf("Test FAILED\n");
	
	//free device memory
	//TODO: use cudaFree();
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	//free host memory
	//TODO: use free();
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}
