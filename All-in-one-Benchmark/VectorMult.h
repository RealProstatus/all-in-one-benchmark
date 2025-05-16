#pragma once

#include <iostream>
#include "Benchmark.h"
#include <chrono>
#include <omp.h>
#include<mkl.h>
#ifndef VECTOR_MULT
#define VECTOR_MULT

class VectorMult:public Benchmark {
public:
    std::string name() const override {
        return "Vectors Multiplication";
    }

    void run() override {
        mkl_set_dynamic(false);
        mkl_set_num_threads(mkl_get_max_threads());
        omp_set_num_threads(mkl_get_max_threads());


        unsigned long N = 1e4;
        double* x4 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y4 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x4 == nullptr || y4 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 10^4) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x4[a] = ((double)rand() / RAND_MAX) * 100;
            y4[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start4 = std::chrono::high_resolution_clock::now();

        double result4 = cblas_ddot(N, x4, 1, y4, 1);

        auto end4 = std::chrono::high_resolution_clock::now();
        auto time4 = std::chrono::duration_cast<std::chrono::microseconds>(end4 - start4);
        double GFLOPS4 = ((N * 2.0) / (time4.count() * 1e3)); // GFLOPS
        double memory4 = (N * 8 * 2) / (time4.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 10^4 SECONDS: " << time4.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^4 GFLOPS: " << GFLOPS4 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^4 MEMORY GB/s: " << memory4 << std::endl;
        std::cout << std::endl;
        mkl_free(x4);
        mkl_free(y4);
        N = N * 10;

        double* x5 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y5 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x5 == nullptr || y5 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 10^4) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x5[a] = ((double)rand() / RAND_MAX) * 100;
            y5[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start5 = std::chrono::high_resolution_clock::now();

        double result5 = cblas_ddot(N, x5, 1, y5, 1);

        auto end5 = std::chrono::high_resolution_clock::now();
        auto time5 = std::chrono::duration_cast<std::chrono::microseconds>(end5 - start5);
        double GFLOPS5 = ((N * 2.0) / (time5.count() * 1e3)); // GFLOPS
        double memory5 = (N * 8 * 2) / (time5.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 10^5 SECONDS: " << time5.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^5 GFLOPS: " << GFLOPS5 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^5 MEMORY GB/s: " << memory5 << std::endl;
        std::cout << std::endl;
        mkl_free(x5);
        mkl_free(y5);
        N = 10 * N;

        double* x6 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y6 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x6 == nullptr || y6 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 10^4) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x6[a] = ((double)rand() / RAND_MAX) * 100;
            y6[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start6 = std::chrono::high_resolution_clock::now();

        double result6 = cblas_ddot(N, x6, 1, y6, 1);

        auto end6 = std::chrono::high_resolution_clock::now();
        auto time6 = std::chrono::duration_cast<std::chrono::microseconds>(end6 - start6);
        double GFLOPS6 = ((N * 2.0) / (time6.count() * 1e3)); // GFLOPS
        double memory6 = (N * 8 * 2) / (time6.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 10^6 SECONDS: " << time6.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^6 GFLOPS: " << GFLOPS6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^6 MEMORY GB/s: " << memory6 << std::endl;
        std::cout << std::endl;

        mkl_free(x6);
        mkl_free(y6);
        N = 10 * N;

        double* x7 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y7 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x7 == nullptr || y7 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 10^4) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x7[a] = ((double)rand() / RAND_MAX) * 100;
            y7[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start7 = std::chrono::high_resolution_clock::now();

        double result7 = cblas_ddot(N, x7, 1, y7, 1);

        auto end7 = std::chrono::high_resolution_clock::now();
        auto time7 = std::chrono::duration_cast<std::chrono::microseconds>(end7 - start7);
        double GFLOPS7 = ((N * 2.0) / (time7.count() * 1e3)); // GFLOPS
        double memory7 = (N * 8 * 2) / (time7.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 10^7 SECONDS: " << time7.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^7 GFLOPS: " << GFLOPS7 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^7 MEMORY GB/s: " << memory7 << std::endl;
        std::cout << std::endl;
        mkl_free(x7);
        mkl_free(y7);
        N = 10 * N;

        double* x8 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y8 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x8 == nullptr || y8 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 10^8) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x8[a] = ((double)rand() / RAND_MAX) * 100;
            y8[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start8 = std::chrono::high_resolution_clock::now();

        double result8 = cblas_ddot(N, x8, 1, y8, 1);

        auto end8 = std::chrono::high_resolution_clock::now();
        auto time8 = std::chrono::duration_cast<std::chrono::microseconds>(end8 - start8);
        double GFLOPS8 = ((N * 2.0) / (time8.count() * 1e3)); // GFLOPS
        double memory8 = (N * 8 * 2) / (time8.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 10^8 SECONDS: " << time8.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^8 GFLOPS: " << GFLOPS8 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 10^8 MEMORY GB/s: " << memory8 << std::endl;
        std::cout << std::endl;
        mkl_free(x8);
        mkl_free(y8);
        N = 2 * N;

        double* x8_2 = (double*)mkl_malloc(N * sizeof(double), 64);
        double* y8_2 = (double*)mkl_malloc(N * sizeof(double), 64);

        if (x8_2 == nullptr || y8_2 == nullptr) {
            std::cerr << "== VECTORS MULTIPLICATION: MEMORY ALLOCATION (FOR 2*10^8) ERROR ==" << std::endl;
        }

        for (unsigned int a = 0; a < N; ++a) {
            x8_2[a] = ((double)rand() / RAND_MAX) * 100;
            y8_2[a] = ((double)rand() / RAND_MAX) * 100;
        }

        auto start9 = std::chrono::high_resolution_clock::now();

        double result9 = cblas_ddot(N, x8_2, 1, y8_2, 1);

        auto end9 = std::chrono::high_resolution_clock::now();
        auto time9 = std::chrono::duration_cast<std::chrono::microseconds>(end9 - start9);
        double GFLOPS9 = ((N * 2.0) / (time9.count() * 1e3)); // GFLOPS
        double memory9 = (N * 8 * 2) / (time9.count() * 1e3); // GB/s
        std::cout << "== VECTORS MULTIPLICATION 2*10^8 SECONDS: " << time9.count() / 1e6 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 2*10^8 GFLOPS: " << GFLOPS9 << std::endl;
        std::cout << "== VECTORS MULTIPLICATION 2*10^8 MEMORY GB/s: " << memory9 << std::endl;
        std::cout << std::endl;
        mkl_free(x8_2);
        mkl_free(y8_2);
    }
};




#endif