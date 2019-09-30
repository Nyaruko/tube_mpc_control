// #include "model.cuh"

namespace mppi_control {

Model::Model(float delta_t, float2* control_constraints) {
    dt_ = delta_t;
    control_rngs_ = new float2[CONTROL_DIM];
    for (int i = 0; i < CONTROL_DIM; i++) {
        control_rngs_[i].x = -FLT_MAX;
        control_rngs_[i].y = FLT_MAX;
    }
    for (int i = 0; i < CONTROL_DIM; i++) {
        control_rngs_[i].x = control_constraints[i].x;
        control_rngs_[i].y = control_constraints[i].y;
    }
    HANDLE_ERROR( cudaMalloc((void**)&control_rngs_d_, CONTROL_DIM*sizeof(float2)) );

    paramsToDevice();
}

void Model::paramsToDevice() {
    HANDLE_ERROR( cudaMemcpy(control_rngs_d_, control_rngs_, CONTROL_DIM*sizeof(float2), cudaMemcpyHostToDevice) );
}

void Model::freeCudaMem() {
    HANDLE_ERROR( cudaFree(control_rngs_d_) );
}

__device__ void Model::enforceConstraints(float* state, float* control, float* d_control) {
    for (int i = 0; i < CONTROL_DIM; i++) {
        if (control[i] < control_rngs_d_[i].x) {
            control[i] = control_rngs_d_[i].x;
            d_control[i] = 0;
        } else if ( control[i] > control_rngs_d_[i].y) {
            control[i] = control_rngs_d_[i].y;
            d_control[i] = 0;
        }
    }
}

__host__ __device__ void Model::computeStateDeriv(float* state, float* control, float* state_der) {
    int i;

    for (i = 0; i < STATE_DIM - CONTROL_DIM; i++) {
        state_der[i] = state[i+3];
    }

    for (i = 0; i < CONTROL_DIM; i++) {
        state_der[i + STATE_DIM - CONTROL_DIM] = control[i];
    }
}

__host__ __device__ void Model::incrementState(float* state, float* state_der) {
    int i;

    for (i = 0; i < STATE_DIM; i++) {
        state[i] += state_der[i] * dt_;
        state_der[i] = 0;
    }
    /* for yaw */
//    state[STATE_DIM - 1]  = wrapPi(state[STATE_DIM - 1]);
    state[STATE_DIM - 1] = - M_PI + fmodf(2.0*M_PI + fmodf(state[STATE_DIM-1] + M_PI, 2.0*M_PI), 2.0*M_PI);
}

}