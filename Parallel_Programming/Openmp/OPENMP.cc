#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>        // OpenMP头文件
#include <immintrin.h>  // SIMD指令头文件（如AVX）

#define N 1024

double A[N][N], b[N], x[N];

// 随机初始化矩阵和向量
void init_matrix(int n) {
    srand(0);
    for (int i = 0; i < n; i++) {
        b[i] = rand() % 100 + 1;
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 100 + 1;
        }
    }
}

// 串行+OpenMP+SIMD高斯消元
void gauss_omp_simd(int n) {
    for (int k = 0; k < n; k++) {
        // 主元归一化
        double pivot = A[k][k];
        // SIMD优化主元归一化
        int j = k;
#ifdef __AVX2__
        __m256d pivot_vec = _mm256_set1_pd(pivot);
        for (; j + 4 <= n; j += 4) {
            __m256d row_vec = _mm256_loadu_pd(&A[k][j]);
            row_vec = _mm256_div_pd(row_vec, pivot_vec);
            _mm256_storeu_pd(&A[k][j], row_vec);
        }
#endif
        for (; j < n; j++) {
            A[k][j] /= pivot;
        }
        b[k] /= pivot;

        // 消元（OpenMP并行）
#pragma omp parallel for schedule(static)
        for (int i = k + 1; i < n; i++) {
            double factor = A[i][k];
            int j = k;
#ifdef __AVX2__
            __m256d factor_vec = _mm256_set1_pd(factor);
            for (; j + 4 <= n; j += 4) {
                __m256d aik_vec = _mm256_loadu_pd(&A[i][j]);
                __m256d akj_vec = _mm256_loadu_pd(&A[k][j]);
                __m256d mul_vec = _mm256_mul_pd(factor_vec, akj_vec);
                __m256d res_vec = _mm256_sub_pd(aik_vec, mul_vec);
                _mm256_storeu_pd(&A[i][j], res_vec);
            }
#endif
            for (; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // 回代（串行）
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
    }
}

int main() {
    int n = N;
    init_matrix(n);

    clock_t start = clock();
    gauss_omp_simd(n);
    clock_t end = clock();

    printf("OpenMP+SIMD Gauss Elimination Time: %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 可选：输出前10个解
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}