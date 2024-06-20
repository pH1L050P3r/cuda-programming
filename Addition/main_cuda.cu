#include<iostream>
#include <cstdlib>
#include <assert.h>
#include "cuda.h"

using namespace std;

void init_arr(int* array, unsigned int size){
	for(int i = 0; i < size; i++)
		array[i] = rand() % 100;	
}

void check_array_sum(int* arr_first, int* arr_second, int* arr_third, unsigned int size){
	for(int i = 0; i < size; i++)
		assert(arr_first[i] + arr_second[i] == arr_third[i]);
}

__global__ void AddVector(int *first, int *second, int *third, int size){
	unsigned int tId = blockIdx.x * blockDim.x + threadIdx.x;
	if(tId < size) third[tId] = first[tId] + second[tId];
}

int main(){
	size_t size = 32;
	int *cpu_first = (int *)malloc(size * sizeof(int));
	int *cpu_second = (int *)malloc(size * sizeof(int));
	int *cpu_third = (int *)malloc(size * sizeof(int));

	init_arr(cpu_first, size);
	init_arr(cpu_second, size);

	int *gpu_first, *gpu_second, *gpu_third;
	cudaMalloc(&gpu_first, size * sizeof(int));
	cudaMalloc(&gpu_second, size * sizeof(int));
	cudaMalloc(&gpu_third, size * sizeof(int));

	cudaMemcpy(gpu_first, cpu_first, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_second, cpu_second, size * sizeof(int), cudaMemcpyHostToDevice);	
	
	int threadPerBlock = 256;
	int blocks = size / threadPerBlock + 1;
	AddVector<<<blocks, threadPerBlock>>>(gpu_first, gpu_second, gpu_third, size);

	cudaMemcpy(cpu_third, gpu_third, size * sizeof(int), cudaMemcpyDeviceToHost);	
	cudaFree(&gpu_first);
	cudaFree(&gpu_second);
	cudaFree(&gpu_third);

	check_array_sum(cpu_first, cpu_second, cpu_third, size);	
	printf("SUCCESS\n");
	return 0;
}
