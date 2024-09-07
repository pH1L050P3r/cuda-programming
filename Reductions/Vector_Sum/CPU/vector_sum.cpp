#include <vector>
#include "vector_sum.hpp"

namespace std{
    const int mod = (1 << 30);
    
    int __sum_vector_cpu(int* vec, size_t size){
        int s = 0;
        for(int i = 0; i < size; i++) s = (s + vec[i]) % mod;
        return s;
    }

    int vector_sum(int* vec, size_t size){
        return __sum_vector_cpu(vec, size);
    }
};