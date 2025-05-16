#include "Runge-Kutt.h"
#include<iostream>
#include<cmath>
#include<chrono>

#include<omp.h>

using namespace std;

std::string RungeKutt::name() const {
    return "Benchmark based on the Runge-Kutta method";
}

void RungeKutt::run() {
    std::cout << "Running custom benchmark..." << std::endl;

    omp_set_num_threads(16);

    const double x_max = 10000.0;
    const double d_x = 1.0e-2;
    const double my_gamma = 1.0;
    const int arr_size = 1 + static_cast<int>(x_max / d_x);
    const double PI = 3.1415926535897932;
    const double d_t = 1.0e-6;
    const double t_max = 1.0;

    double* current = new double[arr_size];
    for (int i = 0; i < arr_size; i++)
        current[i] = 1.0 - cos((2.0 * PI * i * d_x) / x_max);

    double* next = new double[arr_size];
    next[0] = next[arr_size - 1] = 0;

    double* k1 = new double[arr_size];
    double* k2 = new double[arr_size];
    double* k3 = new double[arr_size];
    double* k4 = new double[arr_size];

    k1[0] = k2[0] = k3[0] = k4[0] = 0;
    k1[arr_size - 1] = k2[arr_size - 1] = k3[arr_size - 1] = k4[arr_size - 1] = 0;

    double k = (my_gamma * d_t) / (d_x * d_x);
    double c = d_x / 2.0;
    double rkk = d_x / 6.0;

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int t = 0; t < static_cast<int>(t_max / d_t); t++) {
        for (int i = 1; i < arr_size - 1; i++) {
            k1[i] = k * (current[i - 1] - 2.0 * current[i] + current[i + 1]);
        }
        for (int i = 1; i < arr_size - 1; i++) {
            k2[i] = k * (current[i - 1] + c * k1[i - 1] - 2.0 * (current[i] + c * k1[i]) + current[i + 1] + c * k1[i + 1]);
        }
        for (int i = 1; i < arr_size - 1; i++) {
            k3[i] = k * (current[i - 1] + c * k2[i - 1] - 2.0 * (current[i] + c * k2[i]) + current[i + 1] + c * k2[i + 1]);
        }
        for (int i = 1; i < arr_size - 1; i++) {
            k4[i] = k * (current[i - 1] + d_x * k3[i - 1] - 2.0 * (current[i] + d_x * k3[i]) + current[i + 1] + d_x * k3[i + 1]);
        }
        for (int i = 1; i < arr_size - 1; i++) {
            next[i] = current[i] + rkk * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        std::swap(current, next);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    auto result = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    double seconds = result.count();

    std::cout << "Time: " << seconds << " seconds" << std::endl;
    std::cout << "Performance: " << ((t_max / d_t) * (41 * (arr_size - 2))) / (seconds * 1.0e9) << " GFLOPS" << std::endl;
    std::cout << "Throughput: " << ((t_max / d_t) * (8 * 31 * (arr_size - 2))) / (seconds * pow(2, 30)) << " Gb/s" << std::endl;

    delete[] k1; delete[] k2; delete[] k3; delete[] k4;
    delete[] next;
    delete[] current;
}