#include <iostream>
#include"ADDIntrin.h"
#include"FMAIntrin.h"
#include"MatrixMult.h"
#include"VectorMult.h"
#include "Runge-Kutt.h"
#include "Stencil.h"

int main()
{
    ADDIntrin b1;
    FMAIntrin b2;
    MatrixMult b3;
    VectorMult b4;
    JacobiBenchmark b6;

    b1.run();
    b2.run();
    b3.run();
    b4.run();
    b6.run();
}