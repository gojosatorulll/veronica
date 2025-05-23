#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#define N 1024
#define NUM_THREADS 4

double A[N][N], b[N], x[N];

typedef struct {
    int n;
    int tid;
} thread_param_t;

pthread_barrier_t barrier_div;
pthread_barrier_t barrier_elim;

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

// 静态线程消元线程函数（穿插划分，无SIMD）
void* thread_func(void* arg) {
    thread_param_t* param = (thread_param_t*)arg;
    int n = param->n;
    int tid = param->tid;

    for (int k = 0; k < n; k++) {
        // t0线程做主元归一化
        if (tid == 0) {
            double pivot = A[k][k];
            for (int j = k; j < n; j++)
                A[k][j] /= pivot;
            b[k] /= pivot;
        }

        // 所有线程等待t0完成除法
        pthread_barrier_wait(&barrier_div);

        // 穿插划分消元
        for (int i = k + 1 + tid; i < n; i += NUM_THREADS) {
            double factor = A[i][k];
            for (int j = k; j < n; j++)
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }

        // 所有线程等待消元完成
        pthread_barrier_wait(&barrier_elim);
    }
    return NULL;
}

void gauss_static_threads(int n) {
    pthread_t threads[NUM_THREADS];
    thread_param_t params[NUM_THREADS];

    pthread_barrier_init(&barrier_div, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_elim, NULL, NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; t++) {
        params[t].n = n;
        params[t].tid = t;
        pthread_create(&threads[t], NULL, thread_func, &params[t]);
    }
    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    pthread_barrier_destroy(&barrier_div);
    pthread_barrier_destroy(&barrier_elim);

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
    gauss_static_threads(n);
    clock_t end = clock();

    printf("Static Thread Gauss Elimination (Interleaved Partitioning, non-SIMD) Time: %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }
    return 0;
}