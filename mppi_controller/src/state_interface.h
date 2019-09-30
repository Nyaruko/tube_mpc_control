//
// Created by lhc on 2019/9/28.
//

#ifndef CUDA_TEST_WS_STATE_INTERFACE_H
#define CUDA_TEST_WS_STATE_INTERFACE_H

#include <ros/ros.h>
#include <mutex>
#include "math_static_use.h"
#include "geometry_math_type.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>

class StateInterface{
public:
    StateInterface(ros::NodeHandle& node):nh_(node) {
        pos_sub_ = nh_.subscribe("/vio_data_rigid1/pos", 10, &StateInterface::pos_data_cb, this);
        vel_sub_ = nh_.subscribe("/vio_data_rigid1/vel", 10, &StateInterface::vel_data_cb, this);
        acc_sub_ = nh_.subscribe("/vio_data_rigid1/acc", 10, &StateInterface::acc_data_cb, this);
        att_sub_ = nh_.subscribe("/vio_data_rigid1/att", 10, &StateInterface::att_data_cb, this);
    }

    ~StateInterface() {

    }

    void pos_data_cb(const geometry_msgs::PoseStamped& _data) {
        geometry_msgs::PoseStamped temp_rigid_ = _data;
        state_mutex.lock();
        state_.timestamp = ros::Time::now();//_data.header.stamp;
        state_.pos << _data.pose.position.x, _data.pose.position.y, _data.pose.position.z;
        state_mutex.unlock();
    }

    void vel_data_cb(const geometry_msgs::Vector3Stamped& _data) {
        geometry_msgs::Vector3Stamped temp_vel_ = _data;
        state_mutex.lock();
        state_.vel << _data.vector.x, _data.vector.y, _data.vector.z;
        state_mutex.unlock();
    }

    void acc_data_cb(const geometry_msgs::Vector3Stamped& _data) {
        geometry_msgs::Vector3Stamped temp_acc_ = _data;
        state_mutex.lock();
        state_.acc << _data.vector.x, _data.vector.y, _data.vector.z;
        state_mutex.unlock();
    }

    void att_data_cb(const geometry_msgs::PoseStamped& _data) {
        geometry_msgs::PoseStamped temp_acc_ = _data;
        Eigen::Quaternion<float> temp_q;
        temp_q.w() = _data.pose.orientation.w;
        temp_q.x() = _data.pose.orientation.x;
        temp_q.y() = _data.pose.orientation.y;
        temp_q.z() = _data.pose.orientation.z;
        Eigen::Matrix3f temp_R;
        get_dcm_from_q<float>(temp_R, temp_q);
        state_mutex.lock();
        state_.R = temp_R;
        state_mutex.unlock();
    }

    fullstate_t get_state() {
        state_mutex.lock();
        fullstate_t tmp_state_ = state_;
        state_mutex.unlock();
        return tmp_state_;
    }

private:

    fullstate_t state_;

    ros::NodeHandle nh_;
    ros::Subscriber pos_sub_;
    ros::Subscriber vel_sub_;
    ros::Subscriber acc_sub_;
    ros::Subscriber att_sub_;
    std::mutex state_mutex;
};

#endif //CUDA_TEST_WS_STATE_INTERFACE_H
