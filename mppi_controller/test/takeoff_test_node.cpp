//
// Created by lhc on 2019/9/29.
//

#include "tube_ctrl_srvs/SetArm.h"
#include "tube_ctrl_srvs/SetHover.h"
#include "tube_ctrl_srvs/SetTakeoffLand.h"

#include <ros/ros.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char** argv) {
    ros::init(argc, argv, "mppi_controller" );
    ros::NodeHandle node_("~");

    ros::ServiceClient takeoff_land_srv = node_.serviceClient<tube_ctrl_srvs::SetTakeoffLand>("/takeoff_land");
    ros::ServiceClient hover_set_srv = node_.serviceClient<tube_ctrl_srvs::SetHover>("/hover_pos");

    tube_ctrl_srvs::SetTakeoffLand set_takeoff_land;
    set_takeoff_land.request.takeoff = true;
    set_takeoff_land.request.takeoff_altitude = 1.0f;
    takeoff_land_srv.call(set_takeoff_land);

    ROS_INFO("takeoff test");

    sleep(15);

    tube_ctrl_srvs::SetHover set_hover;
    set_hover.request.x_ned = 8.0;
    set_hover.request.y_ned = 8.0;
    set_hover.request.z_ned = -1.0;
    set_hover.request.yaw = 3.0;
    hover_set_srv.call(set_hover);

    ROS_INFO("hover test");

    sleep(25);


    set_hover.request.x_ned = 0.0;
    set_hover.request.y_ned = 0.0;
    set_hover.request.z_ned = -1.0;
    set_hover.request.yaw = 3.0;
    hover_set_srv.call(set_hover);
    ROS_INFO("hover test");

    sleep(15);

    set_takeoff_land.request.takeoff = false;
    takeoff_land_srv.call(set_takeoff_land);

    return 0;
}
