#include "mppi_controlller/plant.h"

namespace mppi_control {

Plant::Plant(ros::NodeHandle mppi_node, bool debug_mode, int hz):
hz_ = hz,
debug_mode_ = debug_mode {
    numTimesteps_ = getRosParam<int>("num_timesteps", mppi_node);
    deltaT_ = 1.0/hz_; 

    stateSequence_.resize(STATE_DIM * numTimesteps_);
    controlSequence_.resize(CONTROL_DIM * numTimesteps_);

}



}
