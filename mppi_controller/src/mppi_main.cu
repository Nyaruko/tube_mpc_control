#define __CUDACC_VER__ __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__

#include "mppi_controller.cuh"
#include "model.cuh"
#include "simple_cost.cuh"
#include "run_control_loop.cuh"
#include "sim_env.cuh"

#include <ros/ros.h>
#include <atomic>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

    const int MPPI_NUM_ROLLOUTS__ = 2560;
    const int BLOCKSIZE_X = 16;
    using namespace mppi_control;
// namespace mppi_control{


//     typedef Model Model;

    typedef MPPIController<Model, SimpleCosts, MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X> CONTROLLER_T;
// }
    // const int MPPI_NUM_ROLLOUTS__ = 2560;
    // const int BLOCKSIZE_X = 16;


    int main(int argc, char** argv) {

        ros::init(argc, argv, "mppi_controller" );
        ros::NodeHandle node_("~");

        //define the dynamics model for mppi
        float2 control_constraints[4] = {make_float2(-3, 3), make_float2(-3, 3), make_float2(-1, 1), make_float2(-.5, .5)};
        Model* model = new Model(1.0/10.0,  control_constraints);
        Model* sim_model = new Model(1.0/10.0,  control_constraints);
        SimpleCosts* mppi_cost = new SimpleCosts();

        cudaStream_t optimization_stride;// = 1;//getRosParam<int>("optimization_stride", node_);
        cudaStreamCreate(&optimization_stride);

        //define the controller
        float init_u[4] = {0,0,0,0};
        float exploration_std[4] = {3, 3, 1, 0.5};

        MPPIController<Model, SimpleCosts, MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X>* mppi;
        mppi = new MPPIController<Model, SimpleCosts, MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X>(model, mppi_cost, 50, 10, 20.0, exploration_std, init_u, optimization_stride);

        SimEnv<Model>* sim;
        sim = new SimEnv<Model>(sim_model);
        float* sim_state = (float*)malloc(7*sizeof(float));

        Eigen::Matrix<float, CONTROLLER_T::STATE_DIM, 1> state;
        // Eigen::Matrix<float, CONTROLLER_T::CONTROL_DIM, 1> u_res;
        float* u_res = (float*)malloc(CONTROLLER_T::STATE_DIM*sizeof(float));

        // Eigen::Matrix<float, CONTROLLER_T::STATE_DIM, 1> state;
        state << 0,0,0,0,0,0,0,0,0,0;

        sim->init_state(state.data());
        float avg_cost;

        // Eigen::MatrixXf u(4,1);
    // u << 0,0,0,0;
        ros::Rate loop_rate(10);
        while (ros::ok()) {
            ROS_INFO("%.8f", ros::Time::now().toSec());
            mppi->computeControl(state, avg_cost);
            mppi->getControl(u_res);
            // std::cout << u << std::endl;
            ROS_INFO("%.8f", ros::Time::now().toSec());
            sim->update_state(u_res, state);

            loop_rate.sleep();
        }

        return 0;

    }
// }
