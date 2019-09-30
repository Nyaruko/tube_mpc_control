#ifndef RUN_CONTROL_LOOP_CUH_
#define RUN_CONTROL_LOOP_CUH_

#include <eigen3/Eigen/Dense>
namespace mppi_control {
template <class CONTROLLER_T>
void runControlLoop(CONTROLLER_T* controller) {
    Eigen::Matrix<float, CONTROLLER_T::STATE_DIM, 1> state;
    state << 0,0,0,0,0,0,0;

    Eigen::MatrixXf u(4,1);
    u << 0,0,0,0;

    controller->computeControl(state);
}
}
#endif