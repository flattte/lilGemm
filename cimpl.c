#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define m 2048
#define n 2048

void fill_incremental(float** a) { 
    for (size_t i = 0; i < m; ++i) {
        for(size_t j = 0; j < n; ++j) {
            a[i][j] = (float)(i*m + j);
        }
    }
}


void matmul(float** a, float** b, float** c) { 
    for (size_t i = 0; i < n; ++i) { 
        for (size_t j = 0; j < m; ++j) {
            for (size_t k = 0; k < m; ++k){
                c[i][j] = a[i][k] * b[k][j];
            }
        }
    }
}

__attribute__((aligned(64))) float a[m][n];
__attribute__((aligned(64))) float b[m][n];
__attribute__((aligned(64))) float c[m][n];
int main() {
    fill_incremental(a);
    fill_incremental(b);


    return 0;
}