#define __CUDACC_VER__ __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__

#include "tube_controller.cuh"

#include <ros/ros.h>
#include <atomic>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

const int MPPI_NUM_ROLLOUTS__ = 2560;
const int BLOCKSIZE_X = 16;
const int NMPC_FREQ = 10;
const int AC_FREQ = 80;

int main(int argc, char** argv)    {
    ros::init(argc, argv, "tube_mpc_controller");
    ros::NodeHandle node_("~");
    mppi_control::TubeController<MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X, NMPC_FREQ, AC_FREQ>* tube_mpc_controller;
    tube_mpc_controller = new mppi_control::TubeController<MPPI_NUM_ROLLOUTS__, BLOCKSIZE_X, NMPC_FREQ, AC_FREQ>(node_);
    ros::AsyncSpinner spinner(4 /* threads */);
    spinner.start();
    ROS_INFO("tube_ctrl_node start");
    ros::waitForShutdown();
    ROS_INFO("tube_ctrl_node stop");
    spinner.stop();
    return 0;
}
