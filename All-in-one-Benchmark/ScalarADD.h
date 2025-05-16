#pragma once

#include<omp.h>
#include<iostream>

#include"Benchmark.h"

class ScalarADD : public Benchmark {
public:
	std::string name() const override {
		return "Scalar ADD DP";
	}

	void run() override {
        const unsigned long long ITER = 50'000'000'000ULL;
        const double coeff = 0.123456;

        int threads = omp_get_max_threads();
        omp_set_num_threads(threads);

        double t_start = 0.0, t_stop = 0.0;

#pragma omp parallel
        {
            // НАША «scalar ADD» цепочка — 16 независимых double-регистров
            double y0 = 1.0, y1 = 2.0, y2 = 3.0, y3 = 4.0;
            double y4 = 5.0, y5 = 6.0, y6 = 7.0, y7 = 8.0;
            double y8 = 9.0, y9 = 10.0, y10 = 11.0, y11 = 12.0;
            double y12 = 13.0, y13 = 14.0, y14 = 15.0, y15 = 16.0;

            // 1) Спавним потоки и инициализируем цепочки
#pragma omp barrier

// 2) Один поток стартует таймер
#pragma omp single
            t_start = omp_get_wtime();

            // 3) Горячий скалярный loop — 16 независимых ADD на итерацию

#pragma omp for schedule(static)
#pragma novector
            for (unsigned long long i = 0; i < ITER; ++i) {
                y0 += coeff; y1 += coeff;
                y2 += coeff; y3 += coeff;
                y4 += coeff; y5 += coeff;
                y6 += coeff; y7 += coeff;
                y8 += coeff; y9 += coeff;
                y10 += coeff; y11 += coeff;
                y12 += coeff; y13 += coeff;
                y14 += coeff; y15 += coeff;
            }

            // 4) Ждём все потоки, чтобы остановить замер чисто после работы
#pragma omp barrier

#pragma omp single
            t_stop = omp_get_wtime();

            // 5) Защита от оптимизации: «потребляем» y*
            volatile double sink =
                y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7 +
                y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15;
            (void)sink;
        }

        double elapsed = t_stop - t_start;
        unsigned long long total_flop = ITER * 16ULL * threads;
        double gflops = total_flop / (elapsed * 1e9);

        std::cout
            << "Time:   " << elapsed << " s" << std::endl
            << "GFLOPS: " << gflops << std::endl;
	}
};