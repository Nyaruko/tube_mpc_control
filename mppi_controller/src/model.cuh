#ifndef MODEL_CUH_
#define MODEL_CUH_

#include "managed.cuh"
#include "math_static_use.h"
// #include "meta_math.h"
#include "gpu_err_chk.h"
#include <cfloat>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>

namespace mppi_control {

class Model: public Managed {
public:

    static const int STATE_DIM = 10;
    static const int CONTROL_DIM = 4;

    Eigen::Matrix<float, STATE_DIM, 1> state_der_;

    Model(float delta_t, float2* control_constraints);
    ~Model() {}

    void paramsToDevice();

    void freeCudaMem();

    // __device__ void cudaInit(float* theta_s);

    __device__ void enforceConstraints(float* state, float* control, float* d_control);

    // __device__ void computeKinematics(float* state, float* state_der);
  
    __host__ __device__ void computeStateDeriv(float* state, float* control, float* state_der);
  
    __host__ __device__ void incrementState(float* state, float* state_der);
  
    // __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s);

private:
    float dt_;

    //Control constrain
    float2* control_rngs_; // host fields
    float2* control_rngs_d_; // device fields

    //Host fields
    Eigen::Matrix<float, STATE_DIM, 1> state_;

    //Device fields
    float* state_d_;

};

}

#include "model.cu"

#endif