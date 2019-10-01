#ifndef SIMPLE_COST_CUH_
#define SIMPLE_COST_CUH_

#include "managed.cuh"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <cuda_runtime.h>

namespace mppi_control {

// template<class MODEL_T>
class SimpleCosts: public Managed {
    public:
        // static const int STATE_DIM = MODEL_T::STATE_DIM;
        // static const int CONTROL_DIM = MODEL_T::CONTROL_DIM;
        typedef struct costmap_t{
            float res;
            int width;
            int length;
            int height;
            float3 origin;
            costmap_t() {
                res = 0.1;
                width = 100;
                length = 100;
                height = 40;
                origin.x = -1;
                origin.y = -1;
                origin.z = -1;
            }
        } Costmap;

        typedef struct {
            float desired_pos[3];
            float desired_orientation;
            float control_coefficient;
            float speed_coefficient[3];
            float acc_coefficient[3];
            float track_coefficient[4];
            float obstacle_coefficient;
            float obs_dis_coefficient;
            float crash_coefficient;
            Costmap costmap_param;
        } CostParams;


        CostParams params_host_;

        SimpleCosts(ros::NodeHandle _node);

        ~SimpleCosts();

        float getmin_dis(const Eigen::Vector2f& _tmp_pos);

        void allocateCudaMem();

        void freeCudaMem();

        void setNewParams(float* pos_d, float yaw);

        void paramsToDevice();

        void costmapToDevice();

        __device__ float getControlCost(float* u , float* du, float* vars);

        __device__ float getSpeedCost(float* s);

        __device__ float getAccCost(float* s);

        __device__ float getTrackCost(float* s);

        __device__ float getCollisionCost(float* s, int* crash);

        __device__ float computeCost(float* s, float* u, float* du, float* vars, int* crash, int t);

    private:

        std::vector<float> cost_map_host_;
        CostParams* params_d_;
        float* cost_map_d_;
};

}

#include "simple_cost.cu"

#endif