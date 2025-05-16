#define _CRT_SECURE_NO_WARNINGS
#define LOOP_COUNT 5000
#define ACCELERATION_LOOP_COUNT 10
#include "Transpose.h"
#include<iostream>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <chrono>
#include<omp.h>
using namespace std;
using namespace chrono;

std::string Transpose::name() const {
    return "Benchmark based on the Transpose Intel One API method";
}


void Transpose::run() {
    std::cout << "Running custom benchmark..." << std::endl;
    // Установка максимального числа потоков
    int num_procs = omp_get_num_procs();
    omp_set_num_threads(num_procs);
    mkl_set_num_threads(num_procs);
    printf("Running on %d threads\n", num_procs);

    double* A, *AT;
    const int m = 5000, n = 5000;  // Размер матрицы
    const size_t total_elements = m * n;
    const size_t data_size = total_elements * sizeof(double);

    // Выделение памяти с выравниванием
    A = (double*)mkl_malloc(data_size, 64);
    AT = (double*)mkl_malloc(data_size, 64);  // Транспонированная матрица имеет размеры n?m

    if (!A || !AT) {
        fprintf(stderr, "Memory allocation failed\n");
        mkl_free(A);
        mkl_free(AT);
        return EXIT_FAILURE;
    }

    // Параллельная инициализация матрицы
#pragma omp parallel for schedule(static)
    for (int i = 0; i < total_elements; ++i) {
        A[i] = (double)(i + 1);
    }

    //printf("Benchmarking matrix transpose performance\n");
    //printf("Matrix size: %d x %d, %d iterations\n", m, n, LOOP_COUNT);

    // Разогрев
    for (int r = 0; r < ACCELERATION_LOOP_COUNT; ++r) {
        mkl_domatcopy('R', 'T', m, n, 1.0, A, n, AT, m);
    }

    // Основной замер
    auto start = high_resolution_clock::now();

    for (int r = 0; r < LOOP_COUNT; ++r) {
        mkl_domatcopy('R', 'T', m, n, 1.0, A, n, AT, m);
    }

    auto end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(end - start).count();

    // Расчет производительности
    const double total_bytes = 2.0 * LOOP_COUNT * data_size;  // Чтение + запись
    const double total_flops = LOOP_COUNT * total_elements;   // 1 операция на элемент

    const double bandwidth = (total_bytes / (1 << 30)) / total_time;  // GB/s
    const double gflops = (total_flops / 1e9) / total_time;          // GFLOP/s

    printf("\nPerformance results:\n");

    std::cout << "Time: " << total_time << " seconds" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Throughput: " << bandwidth << " Gb/s" << std::endl;

    // Освобождение памяти
    mkl_free(A);
    mkl_free(AT);
}