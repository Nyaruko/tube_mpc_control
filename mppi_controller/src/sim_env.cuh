#ifndef SIM_ENV_CUH
#define SIM_ENV_CUH
#include <eigen3/Eigen/Dense>

namespace mppi_control {
    template<class MODEL_T>
    class SimEnv {
        public:
            static const int STATE_DIM = MODEL_T::STATE_DIM;
            SimEnv(MODEL_T* model) {
                model_ = model;
                state_ = (float*)malloc(STATE_DIM*sizeof(float));
            }
            ~SimEnv() {

            }

            void init_state(float* state) {
                for (int i = 0; i < STATE_DIM; i++) {
                    state_[i] = state[i];
                }
            }

            void update_state(float* u, Eigen::Matrix<float, STATE_DIM, 1>& s) {
                float* s_der;
                s_der = (float*)malloc(STATE_DIM*sizeof(float));
                model_->computeStateDeriv(state_, u, s_der);
                model_->incrementState(state_, s_der);
                for (int i = 0; i < STATE_DIM; i++) {
                    s(i) = state_[i];
                }

                std::cout << "[sim]: state: "  << s.transpose() << std::endl;
            }

        private:
            float* state_;
            MODEL_T* model_;

    };
}

#endif