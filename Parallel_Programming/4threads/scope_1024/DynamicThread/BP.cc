#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 1024
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

// 多线程消元线程函数（块划分）
void* eliminate_thread(void* arg) {
    thread_param_t* param = (thread_param_t*)arg;
    int k = param->k;
    int n = param->n;
    int tid = param->tid;
    int nthreads = param->nthreads;

    int total_rows = n - (k + 1);
    int rows_per_thread = total_rows / nthreads;
    int extra = total_rows % nthreads;
    int start = k + 1 + tid * rows_per_thread + (tid < extra ? tid : extra);
    int end = start + rows_per_thread - 1 + (tid < extra ? 1 : 0);

    for (int i = start; i <= end && i < n; i++) {
        double factor = A[i][k];
        for (int j = k; j < n; j++)
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
        // 主元归一化（单线程）该方法除法与消元过程隔离
        double pivot = A[k][k];
        for (int j = k; j < n; j++)
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

    printf("Gauss Elimination with Dynamic Threads Time: %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 可选：输出前10个解
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}