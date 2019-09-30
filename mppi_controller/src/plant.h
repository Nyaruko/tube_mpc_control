#ifndef PLANT_H_
#define PLANT_H_

#include <ros/ros.h>
#include <vector>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <atomic>

namespace mppi_control {

class Plant {
    public:
        static const int STATE_DIM = 7;
        static const int CONTROL_DIM = 4;
        typedef struct {
            // X-Y-Z position
            float x_pos;
            float y_pos;
            float z_pos;
            // X-Y-Z velocity
            float x_vel;
            float y_vel;
            float z_vel;
            // X-Y-Z accelerate
            // float x_acc;
            // float y_acc;
            // float z_acc;
            // orientation
            float yaw;
            // Quaternions
            // float qw;
            // float qx;
            // float qy;
            // float qz;

        } FullState;

        typedef struct {
            float x_acc;
            float y_acc;
            float z_acc;
            float yaw_rate;
        } Control;

        int numTimesteps_;
        double deltaT_;

        boost::mutex access_guard_;
        
        Plant(ros::NodeHandle mppi_node, bool debug_mode, int hz);
        ~Plant() {}
        Plant::FullState getState();
        std::vector<float> stateSequence_;
        std::vector<float> controlSequence_;

    private:
        FullState full_state_;
        int hz_;
        bool debug_mode_;
};

}
#endif