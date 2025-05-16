#pragma once

#include<iostream>
#include<intrin.h>
#include<chrono>
#include<omp.h>

#include"Benchmark.h"

class ADDIntrin : public Benchmark {
public:
	std::string name() const override {
		return "Intrinsics ADD DP";
	}

	void run() override {
        const unsigned long long ITER = 500000000ULL;
        

        // 10 независимых регистров
        __m256d y0 = _mm256_set1_pd(1.0);
        __m256d y1 = _mm256_set1_pd(2.0);
        __m256d y2 = _mm256_set1_pd(3.0);
        __m256d y3 = _mm256_set1_pd(4.0);
        __m256d y4 = _mm256_set1_pd(5.0);
        __m256d y5 = _mm256_set1_pd(6.0);
        __m256d y6 = _mm256_set1_pd(7.0);
        __m256d y7 = _mm256_set1_pd(8.0);
        __m256d y8 = _mm256_set1_pd(9.0);
        __m256d y9 = _mm256_set1_pd(10.0);

        double start = omp_get_wtime();
        //объявление идет прямо перед pragma затем, чтобы значение coeff точно попало бы в регистр, а не в кеш или память
        const __m256d coeff = _mm256_set1_pd(0.123456);
#pragma omp parallel for
        for (unsigned long long i = 0; i < ITER; ++i) {
            y0 = _mm256_add_pd(y0, coeff);
            y1 = _mm256_add_pd(y1, coeff);
            y2 = _mm256_add_pd(y2, coeff);
            y3 = _mm256_add_pd(y3, coeff);
            y4 = _mm256_add_pd(y4, coeff);
            y5 = _mm256_add_pd(y5, coeff);
            y6 = _mm256_add_pd(y6, coeff);
            y7 = _mm256_add_pd(y7, coeff);
            y8 = _mm256_add_pd(y8, coeff);
            y9 = _mm256_add_pd(y9, coeff);
            //
            y0 = _mm256_add_pd(y0, coeff);
            y1 = _mm256_add_pd(y1, coeff);
            y2 = _mm256_add_pd(y2, coeff);
            y3 = _mm256_add_pd(y3, coeff);
            y4 = _mm256_add_pd(y4, coeff);
            y5 = _mm256_add_pd(y5, coeff);
            y6 = _mm256_add_pd(y6, coeff);
            y7 = _mm256_add_pd(y7, coeff);
            y8 = _mm256_add_pd(y8, coeff);
            y9 = _mm256_add_pd(y9, coeff);
            //
            y0 = _mm256_add_pd(y0, coeff);
            y1 = _mm256_add_pd(y1, coeff);
            y2 = _mm256_add_pd(y2, coeff);
            y3 = _mm256_add_pd(y3, coeff);
            y4 = _mm256_add_pd(y4, coeff);
            y5 = _mm256_add_pd(y5, coeff);
            y6 = _mm256_add_pd(y6, coeff);
            y7 = _mm256_add_pd(y7, coeff);
            y8 = _mm256_add_pd(y8, coeff);
            y9 = _mm256_add_pd(y9, coeff);
        }
        double stop = omp_get_wtime();

        // запоминаем, чтобы код не оптимизировался выкинуть
        double out[40];
        _mm256_storeu_pd(out + 0, y0);
        _mm256_storeu_pd(out + 4, y1);
        _mm256_storeu_pd(out + 8, y2);
        _mm256_storeu_pd(out + 12, y3);
        _mm256_storeu_pd(out + 16, y4);
        _mm256_storeu_pd(out + 20, y5);
        _mm256_storeu_pd(out + 24, y6);
        _mm256_storeu_pd(out + 28, y7);
        _mm256_storeu_pd(out + 32, y8);
        _mm256_storeu_pd(out + 36, y9);

        unsigned long long total_flop = ITER * 30ULL * 4ULL;
        double perf = total_flop / ((stop - start) * 1e9);

        std::cout << "Time: " << (stop - start) << " s" << std::endl << "GFLOPS: " << perf << std::endl;
	}
};
