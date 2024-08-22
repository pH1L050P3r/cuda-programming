#ifndef THREAD_CPU
#define THREAD_CPU

#include <assert.h>

void multiply_matrix_cpu(
    int A_ROW,
    int A_COL,
    int* A,
    int B_ROW,
    int B_COL,
    int* B,
    int OUT_ROW,
    int OUT_COL,
    long long unsigned int* OUT
) {

    assert(A_COL == B_ROW);
    assert(A_ROW == OUT_ROW);
    assert(B_COL == OUT_COL);

    for(unsigned int i = 0; i < OUT_ROW; i++){
        for(unsigned int j = 0; j < OUT_COL; j++){
            long long unsigned int ans = 0;
            for(unsigned int k = 0; k < A_COL; k++){
                ans += A[i * A_COL + k] * B[k * B_COL + j];
            }
            OUT[i * OUT_COL + j] = ans;
        }
    }
    return;
}

#endif