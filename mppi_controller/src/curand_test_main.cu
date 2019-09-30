#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <iostream>


int main(int argc, char** argv) {

    cudaStream_t optimization_stride;// = 1;//getRosParam<int>("optimization_stride", node_);
    cudaStreamCreate(&optimization_stride);

    curandGenerator_t gen_;
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL);

    curandSetStream(gen_,optimization_stride);

    float* du_d_;

    cudaMalloc((void **)& du_d_, 100*sizeof(float));

    curandGenerateNormal(gen_, du_d_, 100, 0.0, 1.0);

    float* du_ = (float*)malloc(100*sizeof(float));

    cudaMemcpy(du_, du_d_, 100*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 100; i++) {
        std::cout << *(du_+i) << std::endl;
    }

    std::cout << std::endl;

}