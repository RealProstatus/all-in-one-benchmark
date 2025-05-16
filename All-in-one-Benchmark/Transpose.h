#pragma once
#include "Benchmark.h"
#include <iostream>

class Transpose : public Benchmark {
public:
    std::string name() const override;
    void run() override;
};