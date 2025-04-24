#include <immintrin.h>
#include <windows.h>
#include <iostream>
#include <cstdlib>
#include <fstream> // 包含文件操作头文件
using namespace std;

const int maxN = 5000;

float** allocMatrix(int rows, int cols) {
    float** mat = new float* [rows];
    for (int i = 0; i < rows; ++i)
        mat[i] = new float[cols];
    return mat;
}

void freeMatrix(float** mat, int rows) {
    for (int i = 0; i < rows; ++i)
        delete[] mat[i];
    delete[] mat;
}

double LU(int n, float** a) {
    LARGE_INTEGER freq, beginTime, endTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&beginTime);

    for (int i = 0; i <= n - 1; i++) {
        for (int j = i + 1; j <= n; j++)
            a[i][j] /= a[i][i];
        a[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            float tem = a[j][i];
            for (int k = i + 1; k <= n; k++)
                a[j][k] -= a[i][k] * tem;
            a[j][i] = 0.0f;
        }
    }

    QueryPerformanceCounter(&endTime);
    return (double)(endTime.QuadPart - beginTime.QuadPart) / freq.QuadPart;
}

double generation(int n, float** a, float* x) {
    LARGE_INTEGER freq, beginTime, endTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&beginTime);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = a[i][n];
        for (int j = i + 1; j < n; j++)
            x[i] -= a[i][j] * x[j];
        x[i] /= a[i][i];
    }

    QueryPerformanceCounter(&endTime);
    return (double)(endTime.QuadPart - beginTime.QuadPart) / freq.QuadPart;
}

double LU_SSE(int n, float** a) {
    LARGE_INTEGER freq, beginTime, endTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&beginTime);
    __m128 t1, t2, sub, tem2;

    for (int i = 0; i <= n - 1; i++) {
        for (int j = i + 1; j <= n; j++)
            a[i][j] /= a[i][i];
        a[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            float tem = a[j][i];
            tem2 = _mm_set1_ps(tem);
            for (int k = i + 1; k + 3 <= n; k += 4) {
                t1 = _mm_loadu_ps(&a[i][k]);
                t2 = _mm_loadu_ps(&a[j][k]);
                sub = _mm_sub_ps(t2, _mm_mul_ps(t1, tem2));
                _mm_storeu_ps(&a[j][k], sub);
            }
            for (int k = n - (n - i) % 4 + 1; k <= n; k++)
                a[j][k] -= a[i][k] * tem;
            a[j][i] = 0.0f;
        }
    }

    QueryPerformanceCounter(&endTime);
    return (double)(endTime.QuadPart - beginTime.QuadPart) / freq.QuadPart;
}

double LU_AVX(int n, float** a) {
    LARGE_INTEGER freq, beginTime, endTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&beginTime);
    __m256 t1, t2, sub, tem2;

    for (int i = 0; i <= n - 1; i++) {
        for (int j = i + 1; j <= n; j++)
            a[i][j] /= a[i][i];
        a[i][i] = 1.0f;
        for (int j = i + 1; j < n; j++) {
            float tem = a[j][i];
            tem2 = _mm256_set1_ps(tem);
            for (int k = i + 1; k + 7 <= n; k += 8) {
                t1 = _mm256_loadu_ps(&a[i][k]);
                t2 = _mm256_loadu_ps(&a[j][k]);
                sub = _mm256_sub_ps(t2, _mm256_mul_ps(t1, tem2));
                _mm256_storeu_ps(&a[j][k], sub);
            }
            for (int k = n - (n - i) % 8 + 1; k <= n; k++)
                a[j][k] -= a[i][k] * tem;
            a[j][i] = 0.0f;
        }
    }

    QueryPerformanceCounter(&endTime);
    return (double)(endTime.QuadPart - beginTime.QuadPart) / freq.QuadPart;
}

int main() {
    int n = 128;
    double result[100][6] = { 0 };

    // 创建并打开 CSV 文件
    ofstream outFile("performance_results.csv");
    if (!outFile.is_open()) {
        cout << "无法打开文件!" << endl;
        return 1;
    }

    // 写入 CSV 文件标题行
    outFile << "Matrix Size,LU Time,LU SSE Time,LU AVX Time,Back Substitution Time,Back Substitution SSE Time,Back Substitution AVX Time" << endl;

    float** a = allocMatrix(maxN, maxN + 1);
    float** b = allocMatrix(maxN, maxN + 1);
    float** c = allocMatrix(maxN, maxN + 1);
    float* answer = new float[maxN];
    float* answer2 = new float[maxN];
    float* answer3 = new float[maxN];

    while (n <= 100) {
        cout << "矩阵规模为 " << n << " × " << n << " 时" << endl;
        for (int x = 0; x < 20; ++x) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= n; ++j) {
                    float val = (rand() % 100000) / 100.0f;
                    a[i][j] = b[i][j] = c[i][j] = val;
                }
                answer[i] = answer2[i] = answer3[i] = 0;
            }

            result[n / 128][0] += LU(n, a);
            result[n / 128][1] += LU_SSE(n, b);
            result[n / 128][2] += LU_AVX(n, c);
            result[n / 128][3] += generation(n, a, answer);
            result[n / 128][4] += generation(n, b, answer2);
            result[n / 128][5] += generation(n, c, answer3);
        }

        for (int j = 0; j < 6; ++j)
            result[n / 128][j] /= 20;

        cout << "普通LU: " << result[n / 128][0]
            << ", SSE: " << result[n / 128][1]
            << ", AVX: " << result[n / 128][2]
            << ", 普通回代: " << result[n / 128][3]
            << ", 回代SSE: " << result[n / 128][4]
            << ", 回代AVX: " << result[n / 128][5] << endl;

        // 写入结果到CSV文件
        outFile << n << ","
            << result[n / 128][0] << ","
            << result[n / 128][1] << ","
            << result[n / 128][2] << ","
            << result[n / 128][3] << ","
            << result[n / 128][4] << ","
            << result[n / 128][5] << endl;

        n += 10;
        cout << endl;
    }

    freeMatrix(a, maxN);
    freeMatrix(b, maxN);
    freeMatrix(c, maxN);
    delete[] answer;
    delete[] answer2;
    delete[] answer3;

    outFile.close(); // 关闭文件
    return 0;
}
