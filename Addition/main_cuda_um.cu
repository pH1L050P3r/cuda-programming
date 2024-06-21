// with unified memory

#include "iostream"
#include "cuda.h"
#include "stdlib.h"
#include "assert.h"

using namespace std;

__global__ void AddVector(int *a, int *b, int *c, size_t count)
{
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < count)
    {
        c[threadId] = a[threadId] + b[threadId];
    }
    return;
}

void init_arr(int* array, unsigned int size){
	for(int i = 0; i < size; i++)
		array[i] = rand() % 100;	
}

void check_array_sum(int* arr_first, int* arr_second, int* arr_third, unsigned int size){
	for(int i = 0; i < size; i++)
		assert(arr_first[i] + arr_second[i] == arr_third[i]);
}

int main(){
    int count = 1000000;
    int device = -1;
    cudaGetDevice(&device);

    int *first, *second, *output;

    // Allocate memory in Unified Memoey way = host and device both can access the memory
    cudaMallocManaged(&first, count * sizeof(int));
    init_arr(first, count);
    
    // prefetching memory into GPU before accessing kernel
    cudaMemPrefetchAsync(first, count * sizeof(int), device, NULL);
    
    cudaMallocManaged(&second, count*sizeof(int));
    init_arr(second, count);
    cudaMemPrefetchAsync(second, count*sizeof(int), device, NULL);

    cudaMallocManaged(&output, count*sizeof(int));
    cudaMemPrefetchAsync(output, count*sizeof(int), device, NULL);

    int block = 256;
    int grid = (count / block) + 1;
    AddVector<<<grid, block>>>(first, second, output, count);

    // for sync. till completion of the AddVector kernel
    cudaDeviceSynchronize();

    check_array_sum(first, second, output, count);	
    printf("SUCCESS\n");

    cudaFree(first);
    cudaFree(second);
    cudaFree(output);

    return 0;  
}
