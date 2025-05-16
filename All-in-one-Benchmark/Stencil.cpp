#include "Stencil.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

JacobiBenchmark::JacobiBenchmark() {
    caches = {
        {"L1", 48 * 1024},
        {"L2", 1250 * 1024},
        {"L3", 24 * 1024 * 1024} // делим на NUM_THREADS позже
    };
}

std::string JacobiBenchmark::name() const {
    return "Jacobi Benchmark";
}

void JacobiBenchmark::run() {
    const int iterations = 100;
    const int numThreads = 4;
    const int totalSize = 10'000'000;
    runAll("jacobi_benchmark_output.csv", iterations, numThreads, totalSize);
}

void JacobiBenchmark::runAll(const std::string& filename, int iterations, int numThreads,
    int totalSize, int jmin, int jmax) {
    std::ofstream fout(filename);
    fout << "jmax,kmax,CacheLevel,jblock,MLUP/s\n";

    omp_set_num_threads(numThreads);

    for (int j = jmin; j <= jmax; j *= 2) {
        int k = (j < 1000) ? j : totalSize / j;
        for (const auto& [label, size] : caches) {
            size_t adjustedSize = label == "L3" ? size / numThreads : size;
            int jblock = jblockFromCache(adjustedSize);
            jblock = std::min(jblock, j - 2);
            double mlups = runSingle(iterations, j, k, jblock);
            fout << j << "," << k << "," << label << "," << jblock << "," << mlups << "\n";
            std::cout << "jmax = " << j << ", kmax = " << k << ", cache = " << label
                << ", jblock = " << jblock << ", MLUP/s = " << mlups << "\n";
        }
    }

    fout.close();
    std::cout << "Results are saved in file: " << filename << "\n";
}

int JacobiBenchmark::jblockFromCache(size_t cacheBytes) {
    return static_cast<int>((cacheBytes / 2) / (3 * sizeof(double)));
}

double JacobiBenchmark::runSingle(int iterations, int jmax, int kmax, int jblock) {
    using Grid = std::vector<std::vector<double>>;
    Grid x(jmax, std::vector<double>(kmax, 1.0));
    Grid y(jmax, std::vector<double>(kmax, 0.0));

    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
#pragma omp parallel for schedule(static)
        for (int jb = 1; jb < jmax - 1; jb += jblock) {
            int jb_end = std::min(jb + jblock, jmax - 1);
            for (int k = 1; k < kmax - 1; ++k) {
                for (int j = jb; j < jb_end; ++j) {
                    y[j][k] = 0.25 * (x[j - 1][k] + x[j + 1][k] + x[j][k - 1] + x[j][k + 1]);
                }
            }
        }
        std::swap(x, y);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double updates = double(jmax - 2) * (kmax - 2) * iterations;
    return updates / (1e6 * seconds);
}
