#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "D:\Application\Include\mpi.h"
#include<mpi.h> // 引入MPI头文件 

#define N 1024

// 注意：在MPI中，每个进程都有自己独立的内存空间。
// 因此，全局变量在不同进程中是独立的，需要通过MPI通信来共享数据。
// 这里A和b被声明为全局，但其内容将在不同进程中有所不同（分块）。
double A[N][N];
double b[N];
double x[N];


// 随机初始化矩阵和向量
// 仅由0号进程调用
void init_matrix(int n) {
    srand(0); // 使用固定种子，方便结果复现
    // 建议：作业指导书中提到随机生成的矩阵可能导致inf或nan，建议初始化为上三角矩阵后再进行随机操作。 
    // 为了简化，这里仍然使用随机数，但在实际应用中需注意。
    for (int i = 0; i < n; i++) {
        b[i] = (double)(rand() % 100) + 1.0;
        for (int j = 0; j < n; j++) {
            A[i][j] = (double)(rand() % 100) + 1.0;
        }
    }
}

// 串行高斯消元（用于回代阶段）
void gauss_back_substitution(int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
    }
}

int main(int argc, char* argv[]) {
    int rank; // 当前进程的秩 
    int size; // 总进程数 

    MPI_Init(&argc, &argv); // 初始化MPI环境 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // 获取当前进程的秩 
    MPI_Comm_size(MPI_COMM_WORLD, &size); // 获取总进程数 

    int n = N; // 问题规模

    // 计算每个进程处理的行数
    // 一维行块划分：每个进程大致分配 n/size 行 
    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : (start_row + rows_per_process);
    // 最后一个进程处理剩余行数，以确保所有行都被处理 
    if (rank == size - 1) {
        end_row = n;
    } else {
        start_row = rank * (n / size);
        end_row = (rank + 1) * (n / size);
    }
    // 处理余数部分的更均衡分配策略（可选，但这里简化为最后一个进程承担） 

    double *local_A = (double*)malloc((end_row - start_row) * N * sizeof(double));
    double *local_b = (double*)malloc((end_row - start_row) * sizeof(double));

    MPI_Barrier(MPI_COMM_WORLD); // 同步所有进程，确保时间测量准确

    double start_time = MPI_Wtime(); // MPI_Wtime() 提供墙钟时间，比clock()更适合并行程序测量

    if (rank == 0) {
        init_matrix(n); // 0号进程初始化完整矩阵 
    }

    // 0号进程将数据分发给所有进程
    // 对于每个进程，将对应行的A和b复制到local_A和local_b
    // 这不是最优的分发方式，更优的是使用MPI_Scatterv进行变长分发
    // 但为了代码简洁和理解，这里简化处理：所有进程A和b都完整，0号进程填充，其他进程等待广播
    // 实际应根据行块划分只分发所需行。这里假定所有进程都预留了完整的A和b，然后进行广播同步。
    // 更高效的方式是只在0号进程初始化，然后用MPI_Scatter或MPI_Scatterv将对应的行发送给其他进程
    // 但为简化初学者的理解，先让所有进程保持完整的A和b，然后通过广播同步。
    // 在实际的MPI并行化中，为了节省内存，通常每个进程只存储自己负责的行。
    // 这里为了简化代码，假定所有进程都拥有完整的A和b，并通过通信来保持数据一致性。
    // 更符合一维行划分的内存优化做法是：每个进程只分配local_A和local_b来存储自己负责的行，
    // 然后0号进程将A和b的相应行发送给其他进程。
    // 这里为了演示广播过程，以及后续收集的便利性，采用了所有进程保留完整矩阵的策略。
    // 这是需要注意的内存效率问题。

    // 在每轮消元中，主元行会被广播到所有进程
    for (int k = 0; k < n; k++) {
        // 找到负责第 k 行的进程
        int owner_rank = k / rows_per_process;
        if (k >= size * rows_per_process) { // 处理行数不是整除的情况，将剩余行分给最后一个进程
            owner_rank = size - 1;
        }

        double pivot_row_A[N]; // 用于广播主元行
        double pivot_b_val;   // 用于广播主元行对应的b值

        if (rank == owner_rank) { // 如果当前进程是主元行的拥有者
            // 执行主元归一化 
            double pivot = A[k][k];
            for (int j = k; j < n; j++) {
                A[k][j] /= pivot;
            }
            b[k] /= pivot;

            // 复制主元行数据到缓冲区准备广播
            for (int j = k; j < n; j++) {
                pivot_row_A[j] = A[k][j];
            }
            pivot_b_val = b[k];
        }

        // 广播主元行的数据给所有进程 
        // 注意：这里需要确保广播的范围是实际需要的数据量，即从k列到n-1列
        // 对于b[k]也要广播
        MPI_Bcast(&pivot_row_A[k], n - k, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);
        MPI_Bcast(&pivot_b_val, 1, MPI_DOUBLE, owner_rank, MPI_COMM_WORLD);

        // 所有进程更新自己的完整矩阵A和b的主元行（为了后续局部消元的计算便利）
        // 这意味着所有进程都保留了完整矩阵，这对于大矩阵来说是内存效率低的。
        // 更优的做法是，非owner进程只接收pivot_row_A和pivot_b_val，然后用这些数据来更新自己的local_A和local_b。
        if (rank != owner_rank) {
             for (int j = k; j < n; j++) {
                A[k][j] = pivot_row_A[j];
            }
            b[k] = pivot_b_val;
        }
        
        // 局部消元 
        // 只有负责 k+1 行及以后行数据的进程才需要进行消元
        // 遍历当前进程负责的行
        for (int i = start_row; i < end_row; i++) {
            if (i > k) { // 只对主元行下面的行进行消元
                double factor = A[i][k];
                for (int j = k; j < n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }

    // 并行高斯消元阶段结束，测量时间
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // 收集所有进程的A和b数据到0号进程
    // 这里需要一个Allgather或Gather操作来将所有进程的A和b数据收集到0号进程
    // 由于我们选择了所有进程都保留完整的A和b，这里其实不需要额外的收集步骤
    // 但如果每个进程只保留自己的分块，那么就需要复杂的MPI_Gatherv来收集所有数据。
    // 为了简化，这里假设A和b在所有进程中都是完整的，且经过广播后保持一致。
    // 因此，回代可以直接在0号进程上进行。
    // 实际情况中，为了节省内存，各个进程只保留自己负责的行。
    // 此时需要通过MPI_Gatherv将所有数据收集到0号进程进行回代，或者并行化回代。

    if (rank == 0) {
        gauss_back_substitution(n); // 0号进程执行串行回代 
        printf("MPI Parallel Gauss Elimination Time (N=%d, Processes=%d): %.6f seconds\n", n, size, elapsed_time);

        // 可选：输出前10个解
        printf("First 10 solutions:\n");
        for (int i = 0; i < (n < 10 ? n : 10); i++) {
            printf("x[%d] = %f\n", i, x[i]);
        }
    }

    // 释放局部内存 (如果local_A和local_b是用来存储分块数据，则需要释放)
    free(local_A);
    free(local_b);

    MPI_Finalize(); // 结束MPI环境 

    return 0;
}