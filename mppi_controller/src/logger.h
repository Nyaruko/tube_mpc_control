//
// Created by lhc on 2019/9/30.
//

#ifndef CUDA_TEST_WS_LOGGER_H
#define CUDA_TEST_WS_LOGGER_H

#include <rosbag/bag.h>
#include <ros/ros.h>
#include <memory>
#include <mutex>
#include <time.h>

class Logger {
public:
    Logger(ros::NodeHandle& node) {
        node.param<std::string>("logger_file_path", _file_path, "/home/lhc/work/CUDA_test_ws/src/mppi_controller/log/");
        is_open = false;
    }

    ~Logger() {
        logger_bag.close();
    }

    void logger_on() {
        std::string _tmp;
        _tmp = _file_path + getTime_string();
        _tmp += ".bag";
        logger_mutex.lock();
        logger_bag.open(_tmp, rosbag::bagmode::Write);
        logger_mutex.unlock();
        std::cout << "logger on: " << _tmp << std::endl;
        is_open = true;
    }

    void logger_off() {
        is_open = false;
        logger_mutex.lock();
        logger_bag.close();
        logger_mutex.unlock();
    }

    template <class T>
    bool logger_write(std::string _name, T _data) {
        if (is_open) {
            logger_mutex.lock();
            logger_bag.write(_name, ros::Time::now(), _data);
            logger_mutex.unlock();
            return true;
        } else {
            return false;
        }
    }

private:

    std::string  getTime_string() {
        time_t timep;
        timep = time(0);
        char tmp[64];
        strftime(tmp, sizeof(tmp), "%Y_%m_%d_%H_%M_%S", localtime(&timep));
        return tmp;
    }

    rosbag::Bag logger_bag;

    bool is_open;

    std::string _file_path;

    std::mutex logger_mutex;
};

#endif //CUDA_TEST_WS_LOGGER_H
