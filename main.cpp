#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <memory.h>

const size_t xs = 2024;
const size_t ys = 2024;

[[clang::optnone]]
void print_gflops(std::chrono::microseconds duration, int m, int p, int n) {
    uint64_t microseconds = duration.count();
    double gflop = static_cast<double>((m*n * (2*p - 1)))/1000000000.0;
    double seconds = static_cast<double>(microseconds / 1000000.0);
    double gflops = gflop / seconds;
    printf("Matmul performed at %lf GFLOPs\n", gflops);
}

void init_mm(float* mm, size_t x, size_t y, float (*f)(size_t i, size_t j)) {
    for(size_t i = 0; i < x; ++i) {
        for(size_t j = 0; j < y; ++j) {
            mm[i*x + j] = f(i, j);
        }
    }
}

float incremental(size_t i, size_t j) {
    return (float)((unsigned int)(i*xs + j + 1));
}

void print(float* mm, size_t x, size_t y, size_t stride=1) {
    printf("[[MAT: m = %lu, n = %lu]], stride=%zu\n", x, y, stride);
    for(size_t i = 0; i < x; i+=stride) {
        for(size_t j = 0; j < y; j+=stride) {
            printf(" %0.3f ", mm[i*x + j]); 
        }
        printf("]");
        printf("\n");
    }
}

void transpose(float* a, float* b, size_t x, size_t y) {
    for(int i = 0; i < x; ++i) {
        for(int j = 0; j < y; ++j) {
            b[i*x + j] = a[j*x + i];
        }
    }
}

__attribute__((aligned(64))) float mm[xs*ys];
__attribute__((aligned(64))) float mm_t[xs* ys];
__attribute__((aligned(64))) float c[xs * ys];

struct matrix { 
    size_t m;
    size_t n;
    float* mat;
};
typedef struct matrix matrix;
float*  __attribute__((assume_aligned(64))) get_alligned_matrix(float* m) { 
    return m; 
}

//assumes transposition for now;
void matmul(matrix a, matrix b, matrix c) { 
    if (a.n != b.m) {
        printf("Matrices cannot be multiplied reason: (m0,n0) . (m1, n1) => n0 != m1");
        return;
    }
    float* A __attribute__((aligned(64))) = get_alligned_matrix(a.mat);
    float* B __attribute__((aligned(64))) = get_alligned_matrix(b.mat);
    float* C __attribute__((aligned(64))) = get_alligned_matrix(c.mat); 

    float acc = 0.0f;
    for (int i = 0; i < a.m; ++i) {
        for (int j = 0; j < b.n; ++j) {
            for (int k = 0; k < a.n; ++k) {
                acc += A[i*a.m + k] + B[j*b.n + k];
            }
            C[i*a.m + j] = acc;
            acc = 0.0f;
        } 
    }
}

int main(){
    const size_t stride = 128;
    // //-O3 makes this ridiculously fast
    //__attribute__((aligned(32))) float* mm = (float*)calloc(xs * ys, sizeof(float));
    size_t matrix_size_bytes = sizeof(float)*xs*ys;
    memset(mm, 0, matrix_size_bytes);
    memset(mm_t, 0, matrix_size_bytes);
    memset(c, 0, matrix_size_bytes);
    init_mm((float*)mm, xs, ys, incremental);
    
    transpose((float*)mm, (float*)mm_t, xs, ys);
    //print((float*)mm, xs, ys, stride);
    //print((float*)mm_t, xs, ys, stride);
    fflush(stdout);

    matrix A = {xs, ys, mm};
    matrix B = {xs, ys, mm_t};
    matrix C = {xs, ys, c};

    auto start_time = std::chrono::high_resolution_clock::now();
    // float acc = 0.0f;

    // for(int i = 0; i < xs; ++i) {
    //     for(int j = 0; j < ys; ++j) {
    //         for(int k = 0; k < xs; ++k) {
    //             acc += mm[i*xs + k] * mm_t[j*xs + k]; 
    //         }
    //         c[i*xs + j] = acc;
    //         acc = 0.0f; 
    //     }
    // }
    
    matmul(A, B, C);


    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Matmul done in %ld microseconds [10^(-6)s]\n", duration.count());
    print_gflops(duration, xs, ys, ys);
    
    print(C.mat, xs, ys, stride);
    return 0;
}



