#include <metal_stdlib>
#include <metal_math>
using namespace metal;

kernel void calculatePi(device float *x [[buffer(0)]],
                        constant int& numOfSteps [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant int& sizeOfBlock [[buffer(3)]],
                        const uint tgPos [[ threadgroup_position_in_grid ]],
                        const uint threadsPerTG [[ threads_per_threadgroup ]],
                        const uint tPos [[ thread_position_in_threadgroup ]])
{
    int yIndex = tgPos * threadsPerTG + tPos;
    int xStart = yIndex * sizeOfBlock;
    int xEnd = xStart + sizeOfBlock < numOfSteps ? xStart + sizeOfBlock : numOfSteps;
    for (int xIndex = xStart; xIndex < xEnd; xIndex++)
    {
        y[yIndex] += sqrt(1.0 - x[xIndex] * x[xIndex]);
    }
}
