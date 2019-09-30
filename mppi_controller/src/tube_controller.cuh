#ifndef TUBE_CONTROLLER_CUH_
#define TUBE_CONTROLLER_CUH_

#include "mppi_controller.cuh"
#include "model.cuh"
#include "simple_cost.cuh"
#include "run_control_loop.cuh"
#include "nominal_system.cuh"
#include "associate_controller_core.cuh"
#include "in_loop_cmd_generator.h"
#include "geometry_math_type.h"
#include "mavros_interface.h"
#include "math_static_use.h"
#include "state_interface.h"
#include "logger.h"

#include "nav_msgs/Odometry.h"
#include "std_msgs/Float32.h"

// srv
#include "tube_ctrl_srvs/SetArm.h"
#include "tube_ctrl_srvs/SetHover.h"
#include "tube_ctrl_srvs/SetTakeoffLand.h"

#include <ros/ros.h>
#include <atomic>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <thread>
#include <mutex>

namespace mppi_control {

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    class TubeController {
    public:
        TubeController(ros::NodeHandle node_);
        ~TubeController();
        void NMPC_thread_handler();
        void AC_thread_handler();
        void NMPC_thread_loop();
        void AC_thread_loop();

    private:
        void arm_disarm_vehicle(const bool& arm);
        void set_hover_pos(const Eigen::Vector3f& pos, const float& yaw);
        bool arm_disarm_srv_handle(tube_ctrl_srvs::SetArm::Request& req,tube_ctrl_srvs::SetArm::Response& res);
        bool hover_pos_srv_handle(tube_ctrl_srvs::SetHover::Request& req,tube_ctrl_srvs::SetHover::Response& res);
        bool takeoff_land_srv_handle(tube_ctrl_srvs::SetTakeoffLand::Request& req,tube_ctrl_srvs::SetTakeoffLand::Response& res);
        ros::NodeHandle nh_;
        std::mutex nominal_MPC_thread_mutex;
        std::mutex associate_ctrl_thread_mutex;

        std::mutex tube_ctrl_mutex;

        Eigen::Matrix<float, Model::STATE_DIM, 1> nominal_state;
        Eigen::Matrix<float, Model::CONTROL_DIM, 1> nominal_u;
//        fullstate_t current_state;

        MPPIController<Model, SimpleCosts, ROLLOUTS, BLOCKSIZE_X>* mppi;
        NominalSys<Model>* nominal_sys;
        Model* model;
        Model* nominal_model;
        SimpleCosts* mppi_cost;

        AssociateCtrlCore* ac_ctrl_core;

        InLoopCmdGen* in_loop_cmd_gen;

        Mavros_Interface* mavros_wrapper;

        StateInterface* state_interface;

        Logger* logger;


        /* publisher */
        ros::Publisher nominal_state_pub;
        ros::Publisher cost_pub;

        /*srv list*/
        ros::ServiceServer arm_srv;
        ros::ServiceServer hoverpos_srv;
        ros::ServiceServer takeoff_land_srv;
    };

}

#include "tube_controller.cu"
#endif
