#pragma once

#include<iostream>
#include<intrin.h>
#include<chrono>
#include<omp.h>

#include"Benchmark.h"

class FMAIntrin : public Benchmark {
public:
	std::string name() const override {
        //название вашего бенча
		return "Intrinsics FMA DP";
	}

	void run() override {
        //код вашего бенча, пусть все результаты он выводит непосредственно в консоль, впоследствии сделаем красивый логгер

		const unsigned long long ITERATIONS = 500000000ULL;

		double a[4], b[4], c[4], d[4], e[4], f[4], g[4], h[4], ij[4], ji[4];

        //the values of these registers do not change
        __m256d x1 = _mm256_set_pd(2.0, 2.0, 2.0, 2.0);
        __m256d x2 = _mm256_setzero_pd();
        __m256d x4 = _mm256_set_pd(-120.0, -60.0, -30.0, -15.0);
        //
        __m256d x0 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x3 = _mm256_setzero_pd();
        //
        __m256d x5 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x6 = _mm256_setzero_pd();
        //
        __m256d x7 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x8 = _mm256_setzero_pd();
        //
        __m256d x9 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x10 = _mm256_setzero_pd();
        //
        __m256d x11 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x12 = _mm256_setzero_pd();
        //
        __m256d x13 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x14 = _mm256_setzero_pd();
        //
        __m256d x15 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x16 = _mm256_setzero_pd();
        //
        __m256d x17 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x18 = _mm256_setzero_pd();
        //
        __m256d x19 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x20 = _mm256_setzero_pd();
        //
        __m256d x21 = _mm256_set_pd(8.0, 4.0, 2.0, 1.0);
        __m256d x22 = _mm256_setzero_pd();

        auto start = omp_get_wtime();
#pragma omp parallel for
        for (unsigned long long i = 0ULL; i < ITERATIONS; i++) {
            //
            x3 = _mm256_fmadd_pd(x0, x1, x2);
            x6 = _mm256_fmadd_pd(x5, x1, x2);
            x8 = _mm256_fmadd_pd(x7, x1, x2);
            x10 = _mm256_fmadd_pd(x9, x1, x2);
            x12 = _mm256_fmadd_pd(x11, x1, x2);
            x14 = _mm256_fmadd_pd(x13, x1, x2);
            x16 = _mm256_fmadd_pd(x15, x1, x2);
            x18 = _mm256_fmadd_pd(x17, x1, x2);
            x20 = _mm256_fmadd_pd(x19, x1, x2);
            x22 = _mm256_fmadd_pd(x21, x1, x2);
            //
            x3 = _mm256_fmadd_pd(x3, x1, x2);
            x6 = _mm256_fmadd_pd(x6, x1, x2);
            x8 = _mm256_fmadd_pd(x8, x1, x2);
            x10 = _mm256_fmadd_pd(x10, x1, x2);
            x12 = _mm256_fmadd_pd(x12, x1, x2);
            x14 = _mm256_fmadd_pd(x14, x1, x2);
            x16 = _mm256_fmadd_pd(x16, x1, x2);
            x18 = _mm256_fmadd_pd(x18, x1, x2);
            x20 = _mm256_fmadd_pd(x20, x1, x2);
            x22 = _mm256_fmadd_pd(x22, x1, x2);
            //
            x3 = _mm256_fmadd_pd(x3, x1, x2);
            x6 = _mm256_fmadd_pd(x6, x1, x2);
            x8 = _mm256_fmadd_pd(x8, x1, x2);
            x10 = _mm256_fmadd_pd(x10, x1, x2);
            x12 = _mm256_fmadd_pd(x12, x1, x2);
            x14 = _mm256_fmadd_pd(x14, x1, x2);
            x16 = _mm256_fmadd_pd(x16, x1, x2);
            x18 = _mm256_fmadd_pd(x18, x1, x2);
            x20 = _mm256_fmadd_pd(x20, x1, x2);
            x22 = _mm256_fmadd_pd(x22, x1, x2);
            //
            x0 = _mm256_fmadd_pd(x3, x1, x4);
            x5 = _mm256_fmadd_pd(x6, x1, x4);
            x7 = _mm256_fmadd_pd(x8, x1, x4);
            x9 = _mm256_fmadd_pd(x10, x1, x4);
            x11 = _mm256_fmadd_pd(x12, x1, x4);
            x13 = _mm256_fmadd_pd(x14, x1, x4);
            x15 = _mm256_fmadd_pd(x16, x1, x4);
            x17 = _mm256_fmadd_pd(x18, x1, x4);
            x19 = _mm256_fmadd_pd(x20, x1, x4);
            x21 = _mm256_fmadd_pd(x22, x1, x4);
        }
        auto stop = omp_get_wtime();

        auto res = stop - start;

        _mm256_storeu_pd(a, x3);
        _mm256_storeu_pd(b, x6);
        _mm256_storeu_pd(c, x8);
        _mm256_storeu_pd(d, x10);
        _mm256_storeu_pd(e, x12);
        _mm256_storeu_pd(f, x14);
        _mm256_storeu_pd(g, x16);
        _mm256_storeu_pd(h, x18);
        _mm256_storeu_pd(ij, x20);
        _mm256_storeu_pd(ji, x22);

        unsigned long long flop = ITERATIONS * 40 * 2 * 4;
        std::cout << "Time: " << res << " seconds;" << std::endl;
        std::cout << "Performance: " << flop / (res * 1.0e9) << " GFLOPS" << std::endl;
	}
};