#ifndef MPPI_CONTROLLER_CUH
#define MPPI_CONTROLLER_CUH

#include "managed.cuh"
// #include "meta_math.h"
#include "gpu_err_chk.h"
#include <eigen3/Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <iostream>

namespace mppi_control {
template<class MODEL_T, class COST_T, int ROLLOUTS = 2560, int BDIM_X = 64>
class MPPIController {
public:
    static const int BLOCKSIZE_WRX = 64;
    static const int NUM_ROLLOUTS = ( ROLLOUTS / BLOCKSIZE_WRX ) * BLOCKSIZE_WRX;
    static const int BLOCKSIZE_X = BDIM_X;
    static const int STATE_DIM = MODEL_T::STATE_DIM;
    static const int CONTROL_DIM = MODEL_T::CONTROL_DIM;

    cudaStream_t stream_;

    int numTimesteps_;
    int hz_;
    
    MODEL_T *model_;
    COST_T *mppi_cost_;

    MPPIController(MODEL_T* model, COST_T* mppi_cost, int num_timesteps, int hz, float gamma,
                   float* exploration_var, float* init_control, cudaStream_t stream);// = 0);
    
    ~MPPIController();

    void setCudaStream(cudaStream_t stream);

    void allocateCudaMem();
    
    void deallocateCudaMem();

    void setCostParams(Eigen::Vector3f pos_d, float yaw_d);

    void computeControl(Eigen::Matrix<float, STATE_DIM, 1> state, float& avg_cost);//, Eigen::Matrix<float, CONTROL_DIM, 1> u_res);

    void getControl(float* u);

    void getControl(Eigen::Matrix<float, CONTROL_DIM, 1>& u);
    // void computeControl(float* state);

private:
    // int num_iters_;
    float gamma_;

    float normalizer_;

    curandGenerator_t gen_;

    std::vector<float> U_;
    std::vector<float> du_;
    std::vector<float> nu_;
    std::vector<float> init_u_;
    std::vector<float> traj_costs_;
    
    std::vector<float> U_res_;

    float* state_d_;
    float* nu_d_;
    float* U_d_;
    float* du_d_;
    float* traj_costs_d_;
};




}

#include "mppi_controller.cu"
#endif