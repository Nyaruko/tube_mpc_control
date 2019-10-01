#include "gpu_err_chk.h"
// #include "debug_kernels.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "math_static_use.h"

namespace mppi_control {

    SimpleCosts::SimpleCosts(ros::NodeHandle _node) {
        // HANDLE_ERROR( cudaMalloc((void**)&params_d_, sizeof(CostParams)) );
        /* param list */
        float _ctrl_c = 1.0;
        float _speed_c[3] = {3.0, 3.0, 3.0};
        float _acc_c[3] = {1.0, 1.0, 1.0};
        float _track_c[4] = {20.0, 20.0, 120.0, 100.0};
        float _obs_c = 350.0;
        float _dis_c = 2.0;
        float _crash_c = 1000.0;
        _node.param<float>("ctrl_coefficient", _ctrl_c, 1.0);
        _node.param<float>("speed_coefficient/x", _speed_c[0], 3.0);
        _node.param<float>("speed_coefficient/y", _speed_c[1], 3.0);
        _node.param<float>("speed_coefficient/z", _speed_c[2], 3.0);
        _node.param<float>("acc_coefficient/x", _acc_c[0], 1.0);
        _node.param<float>("acc_coefficient/y", _acc_c[1], 1.0);
        _node.param<float>("acc_coefficient/z", _acc_c[2], 1.0);
        _node.param<float>("track_coefficient/x", _track_c[0], 20.0);
        _node.param<float>("track_coefficient/y", _track_c[1], 20.0);
        _node.param<float>("track_coefficient/z", _track_c[2], 120.0);
        _node.param<float>("track_coefficient/yaw", _track_c[3], 100.0);
        _node.param<float>("obstacle_coefficient", _obs_c, 350.0);
        _node.param<float>("distance_coefficient", _dis_c, 2.0);
        _node.param<float>("crash_coefficient", _crash_c, 1000.0);
        allocateCudaMem();
        params_host_.desired_pos[0] = 0.0;
        params_host_.desired_pos[1] = 0.0;
        params_host_.desired_pos[2] = -1.0;
        params_host_.desired_orientation = 1.0;
        params_host_.control_coefficient = _ctrl_c;
        params_host_.speed_coefficient[0] = _speed_c[0];
        params_host_.speed_coefficient[1] = _speed_c[1];
        params_host_.speed_coefficient[2] = _speed_c[2];
        params_host_.acc_coefficient[0] = _acc_c[0];
        params_host_.acc_coefficient[1] = _acc_c[1];
        params_host_.acc_coefficient[2] = _acc_c[2];
        params_host_.track_coefficient[0] = _track_c[0];
        params_host_.track_coefficient[1] = _track_c[1];
        params_host_.track_coefficient[2] = _track_c[2];
        params_host_.track_coefficient[3] = _track_c[3];
        params_host_.obstacle_coefficient = _obs_c;
        params_host_.obs_dis_coefficient = _dis_c;
        params_host_.crash_coefficient = _crash_c;

        Costmap _tmp_costmap_param = params_host_.costmap_param;

        // pillars  (2,2)  r = 1

        cost_map_host_.resize(_tmp_costmap_param.width * _tmp_costmap_param.length *_tmp_costmap_param.height);
        for (int i = 0; i < _tmp_costmap_param.width; i++ ) {
            for (int j = 0; j < _tmp_costmap_param.length; j++ ) {
                for (int k = 0; k < _tmp_costmap_param.height; k++ ) {
                    Eigen::Vector2f _tmp_pos(i*_tmp_costmap_param.res + 0.5 + _tmp_costmap_param.origin.x, j*_tmp_costmap_param.res + 0.5 + _tmp_costmap_param.origin.y);//, k*_tmp_costmap_param.res + 0.5);
                    cost_map_host_[i*_tmp_costmap_param.length*_tmp_costmap_param.height + j*_tmp_costmap_param.height + k] = getmin_dis(_tmp_pos);
                }
            }
        }

        paramsToDevice();
        costmapToDevice();
    }

    SimpleCosts::~SimpleCosts() {
    }

    float SimpleCosts::getmin_dis(const Eigen::Vector2f& _tmp_pos) {
        float min_dis = FLT_MAX;
        float _r = 0.2;
        Eigen::Matrix<float, 2, 9> _pillar_pos;
        _pillar_pos.row(0) << 2, 2, 2, 4, 4, 4, 6, 6, 6;
        _pillar_pos.row(1) << 2, 4, 6, 2, 4, 6, 2, 4, 6;
        for (int i = 0; i < 9; i++) {
            float _dis = (_tmp_pos - _pillar_pos.col(i)).norm();
            if (_dis <= _r) {
                _dis = 0;
            } else {
                _dis -= _r;
            }
            if (_dis < min_dis) {
                min_dis = _dis;
            }
        }
        return min_dis;
    }


    void SimpleCosts::allocateCudaMem() {
        HANDLE_ERROR( cudaMalloc((void**)&params_d_, sizeof(CostParams)) );
    }

    void SimpleCosts::freeCudaMem() {
        HANDLE_ERROR(cudaFree(params_d_));
        HANDLE_ERROR(cudaFree(cost_map_d_));
    }

    void SimpleCosts::setNewParams(float* pos_d, float yaw) {
        params_host_.desired_pos[0] = pos_d[0];
        params_host_.desired_pos[1] = pos_d[1];
        params_host_.desired_pos[2] = pos_d[2];
        params_host_.desired_orientation = wrapPi(yaw);
        paramsToDevice();
    }

    void SimpleCosts::paramsToDevice() {
        HANDLE_ERROR( cudaMemcpyAsync(params_d_, &params_host_, sizeof(CostParams), cudaMemcpyHostToDevice, stream_) );
        HANDLE_ERROR( cudaStreamSynchronize(stream_) );   
    }

    void SimpleCosts::costmapToDevice() {
//        HANDLE_ERROR( cudaFree(cost_map_d_));
        HANDLE_ERROR( cudaMalloc((void**)&cost_map_d_, cost_map_host_.size()*sizeof(float)));
        HANDLE_ERROR( cudaMemcpyAsync(cost_map_d_, cost_map_host_.data(), cost_map_host_.size()*sizeof(float), cudaMemcpyHostToDevice, stream_) );
        HANDLE_ERROR( cudaStreamSynchronize(stream_) );
    }

    __device__ float SimpleCosts::getControlCost(float* u , float* du, float* vars) {
        float control_cost = 0;
        for (int i = 0; i < 4; i++) {
            // control_cost += vars[i] *vars[i] *du[i]*du[i] + vars[i]*u[i]*du[i] + 0.5*u[i]*u[i];
            control_cost += du[i]*du[i] + u[i]*du[i] + 0.5*u[i]*u[i];
        }
        return control_cost;
    }


    __device__ float SimpleCosts::getSpeedCost(float* s) {
        float speed_cost = 0;
        speed_cost = params_d_->speed_coefficient[0]*s[3]*s[3] 
                    + params_d_->speed_coefficient[1]*s[4]*s[4]
                    + params_d_->speed_coefficient[2]*s[5]*s[5];
        return speed_cost;
    }

    __device__ float SimpleCosts::getAccCost(float* s) {
        float acc_cost = 0;
        acc_cost = params_d_->acc_coefficient[0]*s[6]*s[6] 
                    + params_d_->acc_coefficient[1]*s[7]*s[7]
                    + params_d_->acc_coefficient[2]*s[8]*s[8];
        return acc_cost;
    }

    __device__ float SimpleCosts::getTrackCost(float* s) {
        float track_cost = 0;
        float delta_x = s[0] - params_d_->desired_pos[0];
        float delta_y = s[1] - params_d_->desired_pos[1];
        float delta_z = s[2] - params_d_->desired_pos[2];
        float delta_orientation = s[9] - params_d_->desired_orientation;
        if (fabsf(delta_orientation) > M_PI) {
            delta_orientation = -(float)((delta_orientation >0.0)-(delta_orientation <0.0))*(2*M_PI - fabsf(delta_orientation));
        }
        track_cost = params_d_->track_coefficient[0]*delta_x*delta_x
                     + params_d_->track_coefficient[1]*delta_y*delta_y
                      + params_d_->track_coefficient[2]*delta_z*delta_z
                       + params_d_->track_coefficient[3]*delta_orientation*delta_orientation;
        return track_cost;
    }

    __device__ float SimpleCosts::getCollisionCost(float* s, int* crash) {
        float crash_cost = params_d_->crash_coefficient;
        if (crash[0] == 1) {
            return crash_cost;
        }
        float collision_cost = 0;
        int i_x = floorf((s[0] - params_d_->costmap_param.origin.x)/params_d_->costmap_param.res);
        int i_y = floorf((s[1] - params_d_->costmap_param.origin.y)/params_d_->costmap_param.res);
        int i_z = floorf((s[2] - params_d_->costmap_param.origin.z)/params_d_->costmap_param.res);
        if ( i_x < 0 || i_x >= params_d_->costmap_param.width) {
            crash[0] = 1;
            return crash_cost;
        }
        if ( i_y < 0 || i_y >= params_d_->costmap_param.length) {
            crash[0] = 1;
            return crash_cost;
        }
        if ( i_z < 0 || i_z >= params_d_->costmap_param.height) {
            crash[0] = 1;
            return crash_cost;
        }

        float _dis = cost_map_d_[i_x*params_d_->costmap_param.length*params_d_->costmap_param.height + i_y*params_d_->costmap_param.height + i_z];
        if (_dis < params_d_->costmap_param.res) {
            crash[0] = 1;
            return crash_cost;
        }

        return params_d_->obstacle_coefficient*exp(-_dis*params_d_->obs_dis_coefficient);

    }

    __device__ float SimpleCosts::computeCost(float* s, float* u, float* du, float* vars, int* crash, int t) {
        float control_cost = getControlCost(u, du, vars);
        float speed_cost = getSpeedCost(s);
        float acc_cost = getAccCost(s);
        float track_cost = getTrackCost(s);
        float collision_cost = getCollisionCost(s, crash);
        
        float cost = params_d_->control_coefficient*control_cost
                        + speed_cost
                        + acc_cost
                        + track_cost
                        + collision_cost;

        if (cost > 1e12 || isnan(cost)) {
            cost = 1e12;
        }
        return cost;
    }

}