
namespace mppi_control {

#define BLOCKSIZE_X_G MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::BLOCKSIZE_X
// #define BLOCKSIZE_Y MPPIController<DYNAMICS_T, COSTS_T, ROLLOUTS, BDIM_X, BDIM_Y>::BLOCKSIZE_Y
#define BLOCKSIZE_WRX_G MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::BLOCKSIZE_WRX
#define STATE_DIM_G MODEL_T::STATE_DIM
#define CONTROL_DIM_G MODEL_T::CONTROL_DIM
// #define SHARED_MEM_REQUEST_GRD DYNAMICS_T::SHARED_MEM_REQUEST_GRD
// #define SHARED_MEM_REQUEST_BLK DYNAMICS_T::SHARED_MEM_REQUEST_BLK
#define NUM_ROLLOUTS_G MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::NUM_ROLLOUTS

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
__global__ void rolloutKernel(int num_timesteps, float* state_d, float* U_d, float* du_d, float* nu_d,
                             float* costs_d, MODEL_T model, COST_T mppi_cost)
{
    int tdx = threadIdx.x;
    // int tdy = threadIdx.y;
    int bdx = blockIdx.x;

    float* s;
    float* s_der;
    float* u;
    float* nu;
    float* du;

    __shared__ float state_shared[BLOCKSIZE_X_G*STATE_DIM_G];
    __shared__ float state_der_shared[BLOCKSIZE_X_G*STATE_DIM_G];
    __shared__ float control_shared[BLOCKSIZE_X_G*CONTROL_DIM_G];
    __shared__ float control_var_shared[BLOCKSIZE_X_G*CONTROL_DIM_G];
    __shared__ float exploration_variance[BLOCKSIZE_X_G*CONTROL_DIM_G];

    float running_cost = 0;


    int global_idx = BLOCKSIZE_X_G*bdx + tdx;
    if (global_idx < NUM_ROLLOUTS_G) {
        // Portion of the shared memery BELONGING TO each thread
        s = &state_shared[tdx*STATE_DIM_G];
        s_der = &state_der_shared[tdx*STATE_DIM_G];
        u = &control_shared[tdx*CONTROL_DIM_G];
        du = &control_var_shared[tdx*CONTROL_DIM_G];
        nu = &exploration_variance[tdx*CONTROL_DIM_G];
        // Load initial state
        for (int i = 0; i < STATE_DIM_G; i++) {
            s[i] = state_d[i];
            s_der[i] = 0;
        }
        // Load nu
        for (int i = 0; i < CONTROL_DIM_G; i++) {
            u[i] = 0;
            du[i] = 0;
            nu[i] = nu_d[i];
        }
    }

    int crash[1];
    crash[0] = 0;

    __syncthreads();

    /*<---- start of simulation loop ----->*/
    for (int i = 0; i < num_timesteps; i++) {
        if (global_idx < NUM_ROLLOUTS_G) {
            for (int j = 0; j < CONTROL_DIM_G; j++) {
                if (global_idx == 0) {
                    du[j] = 0.0;
                    u[j] = U_d[i*CONTROL_DIM_G + j];
                } else if (global_idx >= .99*NUM_ROLLOUTS_G) {
                    /* U_N = rand()*/
                    du[j] = du_d[CONTROL_DIM_G*num_timesteps*(global_idx) + i*CONTROL_DIM_G +j]*nu[j];
                    u[j] = du[j];
                } else {
                    du[j] = du_d[CONTROL_DIM_G*num_timesteps*(global_idx) + i*CONTROL_DIM_G +j]*nu[j];
                    u[j] = U_d[i*CONTROL_DIM_G + j] + du[j];
                }
                du_d[CONTROL_DIM_G*num_timesteps*(global_idx) + i*CONTROL_DIM_G +j] = du[j];
            }
        }

        __syncthreads();
        if (global_idx < NUM_ROLLOUTS_G) {
            model.enforceConstraints(s,u,du);
            for (int j = 0; j < CONTROL_DIM_G; j++) {
                du_d[CONTROL_DIM_G*num_timesteps*(global_idx) + i*CONTROL_DIM_G +j] = du[j];
            }
        }

        __syncthreads();
        if (global_idx < NUM_ROLLOUTS_G && crash[0] == 0) {
            model.computeStateDeriv(s,u,s_der);
        }

        __syncthreads();
        if (global_idx < NUM_ROLLOUTS_G && crash[0] == 0) {
            model.incrementState(s,s_der);
        }

        __syncthreads();
        if (global_idx < NUM_ROLLOUTS_G && i > 0) {
            running_cost += (mppi_cost.computeCost(s, u, du, nu, crash, i) - running_cost)/(i);
        }


        __syncthreads();
        // TODO Crash
    }

//    running_cost /= num_timesteps;

    if (global_idx < NUM_ROLLOUTS_G) {
        costs_d[global_idx] = running_cost;// + mppi_cost.terminalCost(s);
    }
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
__global__ void normExpKernel(float* costs_d, float gamma, float baseline)
{
    int tdx = threadIdx.x;
    int bdx = blockIdx.x;
    if (BLOCKSIZE_X_G*bdx + tdx < ROLLOUTS) {
        float cost_tmp = 0;
        cost_tmp = costs_d[BLOCKSIZE_X_G*bdx + tdx] - baseline;
        costs_d[BLOCKSIZE_X_G*bdx + tdx] = exp( - cost_tmp / gamma);
    }
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
__global__ void weightedUkernel( float* U_d, float* du_d, float* costs_d, float normalizer, int num_steps)
{
    int tdx = threadIdx.x;
    int tdy = threadIdx.y;
    int bdx = blockIdx.x;

    int global_idx = BLOCKSIZE_X_G*bdx + tdx;

    if (global_idx < num_steps) {
        // float U = 0;
        // float sum = 0;
        // float sum_u = 0;
        for (int i = 0; i < ROLLOUTS; i++) {
            // float tmp = exp( -costs_d[i] / gamma);
            // sum += tmp;
            // sum_u += tmp * du_d[i*CONTROL_DIM_G*num_steps+ global_idx * CONTROL_DIM_G + tdy];
            float weight = costs_d[i] / normalizer;
            U_d[global_idx*CONTROL_DIM_G + tdy] += weight*du_d[i*CONTROL_DIM_G*num_steps+ global_idx * CONTROL_DIM_G + tdy];
        }
        // U_d[global_idx*CONTROL_DIM_G + tdy] += sum_u/sum;
    }
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, class COST_T<MODEL_T>, int ROLLOUTS, int BDIM_X>
void launchRolloutKernel(int num_timesteps, float* state_d, float* U_d, float* du_d, float* nu_d, float* costs_d,
                        MODEL_T* model, COST_T* mppi_cost,
                        cudaStream_t stream)
{
    const int GRIDSIZE_X_G = (ROLLOUTS-1)/BLOCKSIZE_X_G + 1;
    dim3 dimBlock(BLOCKSIZE_X_G, 1, 1);
    dim3 dimGrid(GRIDSIZE_X_G, 1, 1);

    rolloutKernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X ><<< dimGrid, dimBlock, 0, stream>>>(num_timesteps, state_d, U_d, du_d, nu_d, costs_d, *model, *mppi_cost);
}


template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
void launchNormExpKernel(float* costs_d, float baseline, float gamma, cudaStream_t stream)
{
    dim3 dimBlock(BLOCKSIZE_X_G, 1, 1);
    dim3 dimGrid((ROLLOUTS-1)/BLOCKSIZE_X_G + 1, 1, 1);
    normExpKernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X><<<dimGrid, dimBlock, 0, stream>>>(costs_d, gamma, baseline);
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
void launchWeightedUKernel(int num_steps, float* U_d, float* du_d, float* costs_d, float normalizer, cudaStream_t stream)
{
    const int GRIDSIZE_X_G = (num_steps-1)/BLOCKSIZE_X_G + 1;
    dim3 dimBlock(BLOCKSIZE_X_G, CONTROL_DIM_G, 1);
    dim3 dimGrid(GRIDSIZE_X_G, 1, 1);

    weightedUkernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X><<< dimGrid, dimBlock, 0, stream>>>(U_d, du_d, costs_d, normalizer, num_steps);
}


// #undef BLOCKSIZE_X
// // #undef BLOCKSIZE_Y
// #undef BLOCKSIZE_WRX
// #undef STATE_DIM
// #undef CONTROL_DIM
// // #undef SHARED_MEM_REQUEST_GRD
// // #undef SHARED_MEM_REQUEST_BLK
// #undef NUM_ROLLOUTS

/******************************************************************************************************************
MPPI Controller implementation
*******************************************************************************************************************/


template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::MPPIController(MODEL_T* model, COST_T* mppi_cost, int num_timesteps, int hz, float gamma,
                   float* exploration_var, float* init_control, cudaStream_t stream)// = 0)
{
    model_ = model;
    mppi_cost_ = mppi_cost;
    curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen_, 1234ULL);

    setCudaStream(stream);


    normalizer_ = 0;

    hz_ = hz;
    numTimesteps_ = num_timesteps;
    gamma_ = gamma;

    nu_.assign(exploration_var, exploration_var + CONTROL_DIM);
    init_u_.assign(init_control, init_control + CONTROL_DIM);
    du_.resize(numTimesteps_ * CONTROL_DIM);
    U_.resize(numTimesteps_ * CONTROL_DIM);
    traj_costs_.resize(NUM_ROLLOUTS);

    U_res_.resize(CONTROL_DIM);

    allocateCudaMem();

    HANDLE_ERROR(cudaMemcpyAsync(nu_d_, nu_.data(), CONTROL_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_));

    cudaStreamSynchronize(stream_);

}


template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::~MPPIController() {
    curandDestroyGenerator(gen_);
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, class COST_T, ROLLOUTS, BDIM_X>::setCudaStream(cudaStream_t stream) {
    stream_ = stream;
    model_->bindToStream(stream_);
    mppi_cost_->bindToStream(stream_);
    curandSetStream(gen_, stream_);
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::allocateCudaMem() {
    HANDLE_ERROR( cudaMalloc((void**)&state_d_, STATE_DIM*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&nu_d_, STATE_DIM*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&traj_costs_d_, NUM_ROLLOUTS*sizeof(float)));

    HANDLE_ERROR( cudaMalloc((void**)&U_d_, CONTROL_DIM*numTimesteps_*sizeof(float)));
    HANDLE_ERROR( cudaMalloc((void**)&du_d_, NUM_ROLLOUTS*CONTROL_DIM*numTimesteps_*sizeof(float)));
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::deallocateCudaMem() {
    cudaFree(state_d_);
    cudaFree(nu_d_);
    cudaFree(traj_costs_d_);
    cudaFree(U_d_);
    cudaFree(du_d_);

    model_->freeCudaMem();

    cudaStreamDestroy(stream_);
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::setCostParams(Eigen::Vector3f pos_d, float yaw_d) {
    mppi_cost_->setNewParams(pos_d.data(), yaw_d);
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::computeControl(Eigen::Matrix<float, STATE_DIM, 1> state, float& avg_cost) {

    model_->paramsToDevice();

    HANDLE_ERROR( cudaMemcpyAsync(state_d_, state.data(), STATE_DIM*sizeof(float), cudaMemcpyHostToDevice, stream_) );
    // HANDLE_ERROR( cudaMemcpy(state_d_, state.data(), STATE_DIM*sizeof(float), cudaMemcpyHostToDevice));//, stream_) );
    // for (int )
        HANDLE_ERROR( cudaMemcpyAsync(U_d_, U_.data(), CONTROL_DIM*numTimesteps_*sizeof(float), cudaMemcpyHostToDevice, stream_) );
        // HANDLE_ERROR( cudaMemcpy(U_d_, U_.data(), CONTROL_DIM*numTimesteps_*sizeof(float), cudaMemcpyHostToDevice));//, stream_) );
        // Generate a bunch of random numbers
        curandGenerateNormal(gen_, du_d_, NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM, 0.0, 1.0);

        // int tmp3 = NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM;

        // float* tmp_res3;
        // tmp_res3 = (float*)malloc(NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM*sizeof(float));
        // cudaMemcpyAsync(tmp_res3, du_d_, tmp3*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        // // cudaMemcpy(tmp_res, du_d_, tmp*sizeof(float), cudaMemcpyDeviceToHost);//, stream_);
        // for (int i = 0; i < 100; i++) {
        //     std::cout << *(tmp_res3 + i) << " ";
        // }
        // std::cout << std::endl;
        // launch the rollout kernel
        launchRolloutKernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X>(numTimesteps_, state_d_, U_d_, du_d_, nu_d_, traj_costs_d_, model_, mppi_cost_, stream_);
        HANDLE_ERROR( cudaMemcpyAsync(traj_costs_.data(), traj_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
        cudaStreamSynchronize(stream_);

        float min_cost = traj_costs_[0];
        float sum_cost = 0;
        for (int i = 0; i < NUM_ROLLOUTS; i++) {
            if (traj_costs_[i] < min_cost) {
                min_cost = traj_costs_[i];
            }
            sum_cost += traj_costs_[i];
        }

        avg_cost = sum_cost/NUM_ROLLOUTS;

        // launch normal kernel

        // int tmp4 = numTimesteps_*CONTROL_DIM;

        // float* tmp_res4;
        // tmp_res4 = (float*)malloc(numTimesteps_*CONTROL_DIM* sizeof(float));
        // cudaMemcpyAsync(tmp_res4, U_d_, tmp4*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        // for (int i = 0; i < numTimesteps_; i++) {
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM ) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +1) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +2) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +3) << std::endl;
        // }
        // std::cout << std::endl;


//        std::cout << "[mppi]: avg cost: " << avg_cost << std::endl;

        launchNormExpKernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X>(traj_costs_d_, min_cost, gamma_, stream_);
        HANDLE_ERROR( cudaMemcpyAsync(traj_costs_.data(), traj_costs_d_, NUM_ROLLOUTS*sizeof(float), cudaMemcpyDeviceToHost, stream_));
        cudaStreamSynchronize(stream_);

        normalizer_ = 0;
        for (int i = 0; i < NUM_ROLLOUTS; i++) {
            normalizer_ += traj_costs_[i];
        }

        launchWeightedUKernel<MODEL_T, COST_T, ROLLOUTS, BDIM_X>(numTimesteps_, U_d_, du_d_, traj_costs_d_, normalizer_, stream_);

        HANDLE_ERROR( cudaMemcpyAsync(U_.data(), U_d_, numTimesteps_*CONTROL_DIM*sizeof(float), cudaMemcpyDeviceToHost, stream_));

        float ctrl_u[CONTROL_DIM];

        for (int i = 0; i < CONTROL_DIM; i++) {
            ctrl_u[i] = U_[i];
        }



        for (int i = 0; i < numTimesteps_-1; i++) {
            for (int j = 0; j < CONTROL_DIM; j++) {
                U_[i*CONTROL_DIM + j] = U_[(i+1)*CONTROL_DIM + j];
            }
        }

        U_[(numTimesteps_-1)*CONTROL_DIM] = 0;
        U_[(numTimesteps_-1)*CONTROL_DIM +1] = 0;
        U_[(numTimesteps_-1)*CONTROL_DIM +2] = 0;
        U_[(numTimesteps_-1)*CONTROL_DIM +3] = 0;
//        std::cout << "[mppi]: ctrl: " << ctrl_u[0] << ", " << ctrl_u[1] << ", " << ctrl_u[2] << ", " << ctrl_u[3] << std::endl;

        for (int i = 0; i < CONTROL_DIM; i++) {
            // res_u(i) = ctrl_u[i];
            U_res_[i] = ctrl_u[i];
        }

        // std::cout << res_u.transpose() << std::endl;
        // int tmp = NUM_ROLLOUTS;
        // float* tmp_res;
        // tmp_res = (float*)malloc( NUM_ROLLOUTS* sizeof(float));
        // cudaMemcpyAsync(tmp_res, traj_costs_d_, tmp*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        // for (int i = 0; i < 100; i++) {
        //     std::cout << *(tmp_res + i) << " ";
        // }
        // std::cout << std::endl;


        // int tmp2 = NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM;

        // float* tmp_res2;
        // tmp_res2 = (float*)malloc(NUM_ROLLOUTS*numTimesteps_*CONTROL_DIM* sizeof(float));
        // cudaMemcpyAsync(tmp_res2, du_d_, tmp2*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        // for (int i = 1000; i < 1100; i++) {
        //     std::cout << *(tmp_res2 + i) << " ";
        // }
        // std::cout << std::endl;

        // int tmp4 = numTimesteps_*CONTROL_DIM;

        // float* tmp_res4;
        // tmp_res4 = (float*)malloc(numTimesteps_*CONTROL_DIM* sizeof(float));
        // cudaMemcpyAsync(tmp_res4, U_d_, tmp4*sizeof(float), cudaMemcpyDeviceToHost, stream_);
        // for (int i = 0; i < numTimesteps_; i++) {
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM ) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +1) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +2) << " ";
        //     std::cout << *( tmp_res4 + i* CONTROL_DIM +3) << std::endl;
        // }
        // std::cout << std::endl;

        cudaStreamSynchronize(stream_);

}


template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::getControl(float* u) {
    for (int i = 0; i < CONTROL_DIM; i++) {
        u[i] = U_res_[i];
    }
}

template<class MODEL_T, class COST_T, int ROLLOUTS, int BDIM_X>
// template<class MODEL_T, int ROLLOUTS, int BDIM_X>
void MPPIController<MODEL_T, COST_T, ROLLOUTS, BDIM_X>::getControl(Eigen::Matrix<float, CONTROL_DIM, 1>& u) {
    for (int i = 0; i < CONTROL_DIM; i++) {
        u(i) = U_res_[i];
    }
}

}
