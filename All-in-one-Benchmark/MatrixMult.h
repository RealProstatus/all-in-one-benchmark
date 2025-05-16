#pragma once

#include <iostream>
#include "Benchmark.h"
#include <chrono>
#include <mkl.h>
#include <omp.h>
#ifndef MATRIX_MULT
#define MATRIX_MULT

class MatrixMult :public Benchmark {
public:
    std::string name() const override {
        return "Matrix Multiplication";
    }

    void run() override {
        unsigned long long N = 16;
        double alpha = 1.0;
        double beta = 0.0;
        while (N <= 8192) {
            double* A = (double*)mkl_malloc(N * N * sizeof(double), 64);
            double* B = (double*)mkl_malloc(N * N * sizeof(double), 64);
            double* C = (double*)mkl_malloc(N * N * sizeof(double), 64);

            if (A == nullptr || B == nullptr || C == nullptr) {
                std::cerr << "Ошибка выделения памяти!" << std::endl;
            }

            for (int i = 0; i < N * N; i++) {
                A[i] = ((double)rand() / RAND_MAX) * 100.0;
                B[i] = ((double)rand() / RAND_MAX) * 100.0;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Умножение матриц: C = alpha * A * B + beta * C
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A, N, B, N, beta, C, N);

            // Остановка таймера
            auto end = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            double memory_traffic = 24 * (N * N);

            std::cout << "Time for " << N << " elements: " << time.count() << " microseconds" << std::endl;
            std::cout << "GFLOPs for " << N << " elements: " << ((2 * N * N * N) / (time.count() * 1e3)) << std::endl;
            std::cout << "Memory traffic: " << memory_traffic / (time.count() * 1e3) << std::endl;
            std::cout << std::endl;
            mkl_free(A);
            mkl_free(B);
            mkl_free(C);
            N = N * 2;
        }
    }
};
#endif