#ifndef NOMINAL_SYS_CUH_
#define NOMINAL_SYS_CUH_
#include <eigen3/Eigen/Dense>

namespace mppi_control {
    template<class MODEL_T>
    class NominalSys{
        public:
            static const int STATE_DIM = MODEL_T::STATE_DIM;
            static const int CTRL_DIM = MODEL_T::CONTROL_DIM;
            NominalSys(MODEL_T* model) {
                model_ = model;
                state_ = (float*)malloc(STATE_DIM*sizeof(float));
                u_ = (float*)malloc(CTRL_DIM*sizeof(float));
            }
            ~NominalSys() {
                delete state_;
                delete u_;
                delete model_;

            }

            void init_state(Eigen::Matrix<float, STATE_DIM, 1> state) {
                for (int i = 0; i < STATE_DIM; i++) {
                    state_[i] = state(i);
                }
            }

            void set_u(Eigen::Matrix<float, CTRL_DIM, 1> u) {
                for (int i = 0; i < CTRL_DIM; i++) {
                    u_[i] = u(i);
                }
            }

            void update_state(Eigen::Matrix<float, STATE_DIM, 1>& s) {
                float* s_der;
                s_der = (float*)malloc(STATE_DIM*sizeof(float));
                model_->computeStateDeriv(state_, u_, s_der);
                model_->incrementState(state_, s_der);
                for (int i = 0; i < STATE_DIM; i++) {
                    s(i) = state_[i];
                }
//                std::cout << "[sim]: state: "  << s.transpose() << std::endl;
            }

        private:
            float* state_;
            float* u_;
            MODEL_T* model_;

    };
}

#endif