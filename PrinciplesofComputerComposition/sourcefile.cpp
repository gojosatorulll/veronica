#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include<chrono>
// 编译执行方式参考：
// 编译， 也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp

// 运行 baseline
// ./outputfile baseline

// 运行 OpenMP
// ./outputfile openmp

// 运行 子块并行优化
// ./outputfile block

// 运行 MPI（假设 4 个进程）
// mpirun -np 4 ./outputfile mpi

// 运行 MPI（假设 4 个进程）
// ./outputfile other


// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致，可以设计其他验证方法，来验证计算的正确性和性能
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) return false;
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            for (int k = 0; k < M; ++k){
                C[i * P + j] += A[i * M + k] * B[k * P + j];
            }
        }
    }
}

// 方式1: 利用OpenMP进行多线程并发的编程 （主要修改函数）
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
                     #pragma omp parallel
    {
        #pragma omp master
        {
            std::cout << "OpenMP线程数: " << omp_get_num_threads() << std::endl;
            std::cout << "OpenMP最大可用线程数: " << omp_get_max_threads() << std::endl;
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0;
            #pragma omp simd
            for (int k = 0; k < M; ++k){
                C[i * P + j] += A[i * M + k] * B[k * P + j];
            }
        }
    }
	std::cout << "matmul_openmp methods..." << std::endl;
}

// 方式2: 利用子块并行思想，进行缓存友好型的并行优化方法 （主要修改函数)
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size = 64) {
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                int i_max = std::min(ii + block_size, N);
                int j_max = std::min(jj + block_size, P);
                int k_max = std::min(kk + block_size, M);
                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double sum = 0;
                        for (int k = kk; k < k_max; ++k) {
                            sum += A[i * M + k] * B[k * P + j];
                        }
                        C[i * P + j] += sum;
                    }
                }
            }
        }
    }
    std::cout << "matmul_block_tiling methods..." << std::endl;
}

// 方式3: 利用MPI消息传递，实现多进程并行优化 （主要修改函数）
void matmul_mpi(int N, int M, int P) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 只主进程初始化A、B
    std::vector<double> A, B;
    if (rank == 0) {
        A.resize(N * M);
        B.resize(M * P);
        init_matrix(A, N, M);
        init_matrix(B, M, P);
    } else {
        B.resize(M * P);
    }

    // 广播B给所有进程
    MPI_Bcast(B.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 行划分
    int rows_per_proc = N / size;
    int remain = N % size;
    int local_rows = rows_per_proc + (rank < remain ? 1 : 0);
    int start_row = rank * rows_per_proc + std::min(rank, remain);

    std::vector<double> local_A(local_rows * M);
    std::vector<double> local_C(local_rows * P, 0);

    // 主进程分发A的行
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            int send_rows = rows_per_proc + (i < remain ? 1 : 0);
            int send_start = i * rows_per_proc + std::min(i, remain);
            MPI_Send(A.data() + send_start * M, send_rows * M, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        std::copy(A.begin(), A.begin() + local_rows * M, local_A.begin());
    } else {
        MPI_Recv(local_A.data(), local_rows * M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 本地计算
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0;
            for (int k = 0; k < M; ++k) {
                sum += local_A[i * M + k] * B[k * P + j];
            }
            local_C[i * P + j] = sum;
        }
    }

    // 主进程收集所有结果
    std::vector<double> C;
    if (rank == 0) C.resize(N * P);
    std::vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int r = rows_per_proc + (i < remain ? 1 : 0);
        recvcounts[i] = r * P;
        displs[i] = (i * rows_per_proc + std::min(i, remain)) * P;
    }
    MPI_Gatherv(local_C.data(), local_rows * P, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[MPI] Done. (仅主进程输出)" << std::endl;
        // 可加性能计时与正确性验证
    }
}

// 方式4: 其他方式 （主要修改函数）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P, int block_size) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> local_A;
    std::vector<double> local_C;
    std::vector<double> B_all(B); // 保证B有数据

    int rows_per_proc = N / size;
    int remain = N % size;
    int local_rows = rows_per_proc + (rank < remain ? 1 : 0);
    int start_row = rank * rows_per_proc + std::min(rank, remain);

    local_A.resize(local_rows * M);
    local_C.resize(local_rows * P, 0);

    // 主进程分发A
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            int send_rows = rows_per_proc + (i < remain ? 1 : 0);
            int send_start = i * rows_per_proc + std::min(i, remain);
            MPI_Send(A.data() + send_start * M, send_rows * M, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
        std::copy(A.begin(), A.begin() + local_rows * M, local_A.begin());
    } else {
        MPI_Recv(local_A.data(), local_rows * M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 广播B
    MPI_Bcast(B_all.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // OpenMP多线程本地计算
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < local_rows; ii += block_size) {
        for (int jj = 0; jj < P; jj += block_size) {
            for (int kk = 0; kk < M; kk += block_size) {
                int i_max = std::min(ii + block_size, local_rows);
                int j_max = std::min(jj + block_size, P);
                int k_max = std::min(kk + block_size, M);
                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double sum = 0;
                        #pragma omp simd
                        for (int k = kk; k < k_max; ++k) {
                            sum += local_A[i * M + k] * B_all[k * P + j];
                        }
                        local_C[i * P + j] += sum;
                    }
                }
            }
        }
    }

    // 主进程收集所有结果
    std::vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        int r = rows_per_proc + (i < remain ? 1 : 0);
        recvcounts[i] = r * P;
        displs[i] = (i * rows_per_proc + std::min(i, remain)) * P;
    }
    MPI_Gatherv(local_C.data(), local_rows * P, MPI_DOUBLE,
                rank == 0 ? C.data() : nullptr, recvcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "[Other] (MPI+OpenMP混合) Done." << std::endl;
    }
}

int main(int argc, char** argv) {
    omp_set_num_threads(12); // 这里设置为12线程，可根据需要修改
    const int N = 1024, M = 2048, P = 512;
    std::string mode = argc >= 2 ? argv[1] : "baseline";

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0);
    std::vector<double> C_ref(N * P, 0);

    init_matrix(A, N, M);
    init_matrix(B, M, P);
    auto t1=std::chrono::high_resolution_clock::now();
    matmul_baseline(A, B, C_ref, N, M, P);
    auto t2=std::chrono::high_resolution_clock::now();
    double baseline_time=std::chrono::duration<double>(t2-t1).count();
if(mode=="baseline"){
    std::cout << "[Baseline] Time: " << baseline_time << "s\n";
}
    if (mode == "mpi") {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        double mpi_time = 0;
        auto t_mpi1 = std::chrono::high_resolution_clock::now();
        matmul_mpi(N, M, P);
        auto t_mpi2 = std::chrono::high_resolution_clock::now();
        mpi_time = std::chrono::duration<double>(t_mpi2 - t_mpi1).count();

        if (rank == 0) {
            std::cout << "[MPI] Time: " << mpi_time << "s\n";
            std::cout << "[MPI] Speedup: " << baseline_time / mpi_time << std::endl;
        }
        MPI_Finalize();
        return 0;
    } else if (mode == "openmp") {
         auto t3 = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        auto t4 = std::chrono::high_resolution_clock::now();
        double openmp_time = std::chrono::duration<double>(t4 - t3).count();
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[OpenMP] Time: " << openmp_time << "s\n";
        std::cout << "[OpenMP] Speedup: " << baseline_time / openmp_time << std::endl;
    } else if (mode == "block") {
        auto t5=std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P);
        auto t6=std::chrono::high_resolution_clock::now();
        double block_time = std::chrono::duration<double>(t6 - t5).count();
        std::cout << "[Block] Time: " << block_time << "s\n";
        std::cout << "[Block] Valid: " << validate(C, C_ref, N, P) << std::endl;
        std::cout << "[Block] Speedup: " << baseline_time / block_time << std::endl;
    } else if (mode == "other") {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int best_block = 64;
    double best_time = 1e9;
    std::vector<int> block_sizes = {32, 48, 64, 96, 128, 160};
    for (int block_size : block_sizes) {
        MPI_Barrier(MPI_COMM_WORLD);
        std::fill(C.begin(), C.end(), 0.0); // 建议加这一行
        auto t1 = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P, block_size);
        auto t2 = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(t2 - t1).count();
        double t_max;
        MPI_Reduce(&t, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && t_max < best_time) {
            best_time = t_max;
            best_block = block_size;
        }
    }
    if (rank == 0) {
        std::cout << "[Other] Best block_size: " << best_block << ", time: " << best_time << "s\n";
        std::cout << "[Other] Speedup: " << baseline_time / best_time << std::endl;
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl;
    }
    MPI_Finalize();
    return 0;
}
    else {
        std::cerr << "Usage: ./main [baseline|openmp|block|mpi]" << std::endl;
    }
	// 需额外增加性能评测代码或其他工具进行评测
    return 0;
}
