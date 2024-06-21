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

    int *first, *second, *output;
    first = new int[count];
    second = new int[count];
    output = new int[count];

    init_arr(first, count);
    init_arr(second, count);

    int *d_first, *d_second, *d_output;
    if(cudaMalloc(&d_first, sizeof(int) * count) != cudaSuccess){
        cout << "Unable to allocate Memory GPU" << endl;
        return -1;
    }
    if(cudaMalloc(&d_second, sizeof(int)*count) != cudaSuccess){
        cudaFree(d_first);
        cout  << "Unable to allocate Memory in GPU" << endl;

        return -1;
    }
    if(cudaMalloc(&d_output, sizeof(int)*count) != cudaSuccess){
        cudaFree(d_first);
        cudaFree(d_second);

        cout << "Unable to allocate Memory in GPU" << endl;
        return -1;
    }


    //Copying Host to memory    

    if(cudaMemcpy(d_first, first, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
       cudaFree(d_first);
       cudaFree(d_second);
       cudaFree(d_output);

       cout << "Unable to copy from HOST memory to Device Memory"; 
       return -1;
    }

    if(cudaMemcpy(d_second, second, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
       cudaFree(d_first);
       cudaFree(d_second);
       cudaFree(d_output);

       cout << "Unable to copy from HOST memory to Device Memory"; 

       return -1;
    }

    int block = 256;
    int grid = (count / block) + 1;
    AddVector<<<grid, block>>>(d_first, d_second, d_output, count);

    if(cudaMemcpy(output, d_output, sizeof(int)*count, cudaMemcpyDeviceToHost) != cudaSuccess){
	cudaFree(d_first);
       	cudaFree(d_second);
       	cudaFree(d_output);

       	cout << "Kernel called but unable to copy memory from device to host"; 
       	return -1;
    }

    check_array_sum(first, second, output, count);	
    printf("SUCCESS\n");

    delete[] first;
    delete[] second;
    delete[] output;


    return 0;  
}
