#pragma once

#include <string>
#include <vector>
#include <utility>
#include "Benchmark.h"

class JacobiBenchmark : public Benchmark {
public:
    JacobiBenchmark();

    std::string name() const override;
    void run() override;

private:
    std::vector<std::pair<std::string, size_t>> caches;

    void runAll(const std::string& filename, int iterations, int numThreads, int totalSize,
        int jmin = 100, int jmax = 128000);
    int jblockFromCache(size_t cacheBytes);
    double runSingle(int iterations, int jmax, int kmax, int jblock);
};
