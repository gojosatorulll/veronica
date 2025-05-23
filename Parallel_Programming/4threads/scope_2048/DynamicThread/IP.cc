#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h> // 引入AVX指令集头文件

#define N 2048
#define NUM_THREADS 4 // 可根据实际CPU核数调整

double A[N][N], b[N], x[N];

typedef struct {
    int k;
    int n;
    int tid;
    int nthreads;
} thread_param_t;

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

// 多线程消元线程函数（穿插划分+AVX优化）
void* eliminate_thread(void* arg) {
    thread_param_t* param = (thread_param_t*)arg;
    int k = param->k;
    int n = param->n;
    int tid = param->tid;
    int nthreads = param->nthreads;

    for (int i = k + 1 + tid; i < n; i += nthreads) {
        double factor = A[i][k];
        __m256d factor_vec = _mm256_set1_pd(factor);
        int j = k;
        // 使用AVX一次处理4个double
        for (; j + 3 < n; j += 4) {
            __m256d aik_vec = _mm256_loadu_pd(&A[k][j]);
            __m256d aij_vec = _mm256_loadu_pd(&A[i][j]);
            __m256d mul_vec = _mm256_mul_pd(aik_vec, factor_vec);
            aij_vec = _mm256_sub_pd(aij_vec, mul_vec);
            _mm256_storeu_pd(&A[i][j], aij_vec);
        }
        // 处理剩余元素
        for (; j < n; j++)
            A[i][j] -= factor * A[k][j];
        b[i] -= factor * b[k];
    }
    return NULL;
}

// 动态线程高斯消元
void gauss_dynamic_threads(int n) {
    pthread_t threads[NUM_THREADS];
    thread_param_t params[NUM_THREADS];

    for (int k = 0; k < n; k++) {
        // 主元归一化（单线程+AVX优化）
        double pivot = A[k][k];
        __m256d pivot_vec = _mm256_set1_pd(pivot);
        int j = k;
        for (; j + 3 < n; j += 4) {
            __m256d row_vec = _mm256_loadu_pd(&A[k][j]);
            row_vec = _mm256_div_pd(row_vec, pivot_vec);
            _mm256_storeu_pd(&A[k][j], row_vec);
        }
        for (; j < n; j++)
            A[k][j] /= pivot;
        b[k] /= pivot;

        // 多线程消元
        for (int t = 0; t < NUM_THREADS; t++) {
            params[t].k = k;
            params[t].n = n;
            params[t].tid = t;
            params[t].nthreads = NUM_THREADS;
            pthread_create(&threads[t], NULL, eliminate_thread, &params[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
    }

    // 回代（单线程）
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
    gauss_dynamic_threads(n);
    clock_t end = clock();

    printf("Gauss Elimination with Dynamic Threads + AVX SIMD Time: %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 可选：输出前10个解
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}