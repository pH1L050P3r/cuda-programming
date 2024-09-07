#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include <chrono>

#include "CPU/vector_sum.hpp"
#include "GPU/naive.h"

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()


const int mod = 1 << 30;

void vector_init(int* vec, size_t size){
    for(int i = 0; i < size; i++) vec[i] = rand() % 1000;
}

void check_sum(int* vec, size_t size, long long int sum){
    int _sum = 0;
    for(int i = 0; i < size; i++) _sum = (_sum + vec[i]) % mod;

    assert(_sum == sum);
}

int main(){

    size_t size = (1 << 29);
    int* vec = (int *)malloc(sizeof(int) * size);

    vector_init(vec, size);

    auto begin = TIME_NOW;  
    int sum = std::vector_sum(vec, size);
    auto end = TIME_NOW;
    std::cout << "CPU execution time: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";  
    check_sum(vec, size, sum);


    begin = TIME_NOW;  
    std::vector_sum_gpu(vec, size, &sum);
    end = TIME_NOW;
    std::cout << "GPU execution time : naive : " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";  
    check_sum(vec, size, sum);
    

    return 0;
}