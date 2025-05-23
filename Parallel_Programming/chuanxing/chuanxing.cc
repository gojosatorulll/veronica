#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

// 串行高斯消元
void gauss_serial(int n) {
    for (int k = 0; k < n; k++) {
        // 主元归一化
        double pivot = A[k][k];
        for (int j = k; j < n; j++)//第二重循环
            A[k][j] /= pivot;
        b[k] /= pivot;

        // 消元
        for (int i = k + 1; i < n; i++) {//第二重循环
            double factor = A[i][k];
            for (int j = k; j < n; j++)//第三重循环
                A[i][j] -= factor * A[k][j];
            b[i] -= factor * b[k];
        }
    }

    // 回代
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
    gauss_serial(n);
    clock_t end = clock();

    printf("Serial Gauss Elimination Time: %.6f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 可选：输出前10个解
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }

    return 0;
}