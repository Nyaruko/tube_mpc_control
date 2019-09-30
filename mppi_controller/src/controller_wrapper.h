#ifndef CONTROLLER_WRAPPER
#define CONTROLLER_WRAPPER
#include <ros/ros.h>

namespace mppi_control {
    class ControlWrapper {
        public:
            ControlWrapper(ros::NodeHandle node);
            ~ControlWrapper();
        private:
            ros::Publisher
    };
}

#endif