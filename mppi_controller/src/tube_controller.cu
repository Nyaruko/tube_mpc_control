namespace mppi_control {

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::TubeController(ros::NodeHandle node_) {
        nh_ = node_;

        /* param list */
        int _num_timestep = 50;
        float _gamma = 18.0;
        float _u_constrain[4] = {10, 10, 5, 1};
        float _u_exploration[4] = {0.3, 0.3, 0.3, 0.05};
        nh_.param<int>("num_timestep", _num_timestep, 50);
        nh_.param<float>("gamma", _gamma, 18.0);
        nh_.param<float>("u_constrain/x", _u_constrain[0], 10.0);
        nh_.param<float>("u_constrain/y", _u_constrain[1], 10.0);
        nh_.param<float>("u_constrain/z", _u_constrain[2], 5.0);
        nh_.param<float>("u_constrain/yaw", _u_constrain[3], 1.0);
        nh_.param<float>("u_exploration/x", _u_exploration[0], 0.3);
        nh_.param<float>("u_exploration/y", _u_exploration[1], 0.3);
        nh_.param<float>("u_exploration/z", _u_exploration[2], 0.3);
        nh_.param<float>("u_exploration/yaw", _u_exploration[3], 0.05);

        float2 contrl_constraints[Model::CONTROL_DIM] = {make_float2(-_u_constrain[0], _u_constrain[0]),
                                                         make_float2(-_u_constrain[1], _u_constrain[1]),
                                                         make_float2(-_u_constrain[2], _u_constrain[2]),
                                                         make_float2(-_u_constrain[3], _u_constrain[3])};
        model = new Model(1.0/(float)NMPC_FREQ,  contrl_constraints);
        nominal_model = new Model(1.0/(float)CTRL_FREQ,  contrl_constraints);
        mppi_cost = new SimpleCosts(node_);

        ac_ctrl_core = new AssociateCtrlCore(node_, (float)CTRL_FREQ);

        in_loop_cmd_gen = new InLoopCmdGen();

        mavros_wrapper = new Mavros_Interface(1);

        state_interface = new StateInterface(node_);

        logger = new Logger(node_);

        cudaStream_t optimization_stride;

        cudaStreamCreate(& optimization_stride);

        float init_u[Model::CONTROL_DIM] = {0,0,0,0};
        float exploration_std[Model::CONTROL_DIM] = {_u_exploration[0], _u_exploration[1], _u_exploration[2], _u_exploration[3]};

        mppi = new MPPIController<Model, SimpleCosts, ROLLOUTS, BLOCKSIZE_X>(model, mppi_cost, _num_timestep, NMPC_FREQ, _gamma, exploration_std, init_u, optimization_stride);

        nominal_sys = new NominalSys<Model>(nominal_model);

        ROS_INFO("waite for state update");
        sleep(2);

        ros::Rate _loop(10);
        while (ros::ok()) {
            fullstate_t current_state = state_interface->get_state();
            if (ros::Time::now() - current_state.timestamp < ros::Duration(0.5)) {
                Eigen::Vector3f _euler_tmp;
                get_euler_from_R<float>(_euler_tmp, current_state.R);
                nominal_state << current_state.pos, current_state.vel, current_state.acc, _euler_tmp(2);
                mppi->setCostParams(current_state.pos, _euler_tmp(2));
                std::cout << "[nominal sys]init: " << nominal_state.transpose() << std::endl;
                break;
            } else {
                ROS_INFO_THROTTLE(1, "state time out, wait for state for initial");
            }
            ros::spinOnce();
            _loop.sleep();
        }

        nominal_u.setZero();

        nominal_sys->init_state(nominal_state);
        nominal_sys->set_u(nominal_u);

        nominal_state_pub = nh_.advertise<nav_msgs::Odometry>("nominal_state", 10);
        cost_pub = nh_.advertise<std_msgs::Float32>("cost", 10);

        arm_srv = nh_.advertiseService("/arm_disarm", &TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::arm_disarm_srv_handle, this);
        hoverpos_srv = nh_.advertiseService("/hover_pos", &TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::hover_pos_srv_handle, this);
        takeoff_land_srv = nh_.advertiseService("/takeoff_land", &TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::takeoff_land_srv_handle, this);

        NMPC_thread_handler();
        AC_thread_handler();

    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::~TubeController() {
        delete mppi;
        delete nominal_sys;
        delete model;
        delete nominal_model;
        delete mppi_cost;
        delete ac_ctrl_core;
        delete in_loop_cmd_gen;
        delete mavros_wrapper;
        delete state_interface;
        delete logger;
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::NMPC_thread_handler() {
        if (nominal_MPC_thread_mutex.try_lock()) {
            std::cout << "[Tube-MPC]: mppi thread start" << std::endl;
            std::thread NMPC_thread(&TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::NMPC_thread_loop, this);
            if (NMPC_thread.joinable()) {
                NMPC_thread.detach();
            }
        } else {
            std::cout << "[Tube-MPC]: mppi thread has started" << std::endl;
        }
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::AC_thread_handler() {
        if (associate_ctrl_thread_mutex.try_lock()) {
            std::cout << "[Tube-MPC]: associate thread start" << std::endl;
            std::thread AC_thread(&TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::AC_thread_loop, this);
            if (AC_thread.joinable()) {
                AC_thread.detach();
            }
        } else {
            std::cout << "[Tube-MPC]: associate thread has started" << std::endl;
        }
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::NMPC_thread_loop() {
        std::cout << "[mppi] thread start" << std::endl;
        ros::Rate loop_rate(NMPC_FREQ);
        while (ros::ok()) {
            ros::Time t1 = ros::Time::now();
            if (mavros_wrapper->state_check()) {
                tube_ctrl_mutex.lock();
                Eigen::Matrix<float, Model::STATE_DIM, 1> tmp_state = nominal_state;
                Eigen::Matrix<float, Model::CONTROL_DIM, 1> tmp_u = nominal_u;
                tube_ctrl_mutex.unlock();
                float avg_cost_;
                mppi->computeControl(tmp_state, avg_cost_);
                std_msgs::Float32 tmp_msgs;
                tmp_msgs.data = avg_cost_;
//                cost_pub.publish(tmp_msgs);
                logger->logger_write<std_msgs::Float32>("nominal_cost",tmp_msgs);
                mppi->getControl(tmp_u);
                tube_ctrl_mutex.lock();
                nominal_u = tmp_u;
                nominal_sys->set_u(tmp_u);
                tube_ctrl_mutex.unlock();
            } else {
                fullstate_t current_state = state_interface->get_state();
                tube_ctrl_mutex.lock();
                if (ros::Time::now() - current_state.timestamp < ros::Duration(1.5)) {
                    Eigen::Vector3f _euler_tmp;
                    get_euler_from_R<float>(_euler_tmp, current_state.R);
                    nominal_state << current_state.pos, current_state.vel, current_state.acc, _euler_tmp(2);
                    nominal_u.setZero();
                    nominal_sys->set_u(nominal_u);
                    mppi->setCostParams(current_state.pos, _euler_tmp(2));
                }
                tube_ctrl_mutex.unlock();
            }
//            std::cout << "[nominal_controller] tatol time: " << 1000.0f*(ros::Time::now()-t1).toSec() << "ms" << std::endl;
            loop_rate.sleep();
        }
        nominal_MPC_thread_mutex.unlock();
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::AC_thread_loop() {
        std::cout << "[ac] thread start" << std::endl;
        ros::Rate loop_rate(CTRL_FREQ);
        while (ros::ok()) {
            /* update the nominal model */
            tube_ctrl_mutex.lock();
            Eigen::Matrix<float, Model::STATE_DIM, 1> tmp_state = nominal_state;
            Eigen::Matrix<float, Model::CONTROL_DIM, 1> tmp_u = nominal_u;
            if (mavros_wrapper->state_check()) {
                nominal_sys->update_state(tmp_state);
                nominal_state = tmp_state;
            }
            tube_ctrl_mutex.unlock();

            nav_msgs::Odometry tmp_msg;
            tmp_msg.header.stamp =  ros::Time::now();
            tmp_msg.pose.pose.position.x = tmp_state(0);
            tmp_msg.pose.pose.position.y = tmp_state(1);
            tmp_msg.pose.pose.position.z = tmp_state(2);
            tmp_msg.pose.pose.orientation.w = tmp_state(9);
            tmp_msg.twist.twist.linear.x = tmp_state(3);
            tmp_msg.twist.twist.linear.y = tmp_state(4);
            tmp_msg.twist.twist.linear.z = tmp_state(5);
//            nominal_state_pub.publish(tmp_msg);
            logger->logger_write<nav_msgs::Odometry>("nominal_state", tmp_msg);

            /* associate control part */
            fullstate_t current_state = state_interface->get_state();
                /* record state */
                nav_msgs::Odometry _record_state;
                _record_state.header.stamp = current_state.timestamp;
                _record_state.pose.pose.position.x = current_state.pos(0);
                _record_state.pose.pose.position.y = current_state.pos(1);
                _record_state.pose.pose.position.z = current_state.pos(2);
                _record_state.twist.twist.linear.x = current_state.vel(0);
                _record_state.twist.twist.linear.y = current_state.vel(1);
                _record_state.twist.twist.linear.z = current_state.vel(2);
                logger->logger_write<nav_msgs::Odometry>("state", _record_state);
            if (ros::Time::now() - current_state.timestamp < ros::Duration(1.5)) {
                Eigen::Matrix<float, Model::STATE_DIM, 1> full_state;
                Eigen::Vector3f _euler_tmp;
                get_euler_from_R<float>(_euler_tmp, current_state.R);
                full_state << current_state.pos, current_state.vel, current_state.acc, _euler_tmp(2);

                if (!mavros_wrapper->state_check()) {
                    ac_ctrl_core->reset_ctrl();
//                    ROS_INFO("reset ac contrl");
                }

                Eigen::Vector3f u_res;
//                std::cout << "nominal_u: " << tmp_u.transpose() << std::endl;
                ac_ctrl_core->cal_u(full_state.data(), tmp_state.data(), tmp_u.data(), u_res);
//                std::cout << "u_res: "<< u_res.transpose() << std::endl;

                InLoopCmdGen::drone_cmd_t _cmd_res =  in_loop_cmd_gen->cal_R_T(u_res, current_state.R, tmp_state(9));
//                std::cout << "tmp9: " << tmp_state(9) << std::endl;

                Eigen::Quaternion<float> cmd_q;
                get_q_from_dcm<float>(cmd_q, _cmd_res.R);
                mavros_wrapper->pub_att_thrust_cmd<float>(cmd_q, _cmd_res.T);
            } else {
                ROS_INFO("[ac ctrl]: state timeout");
            }

            loop_rate.sleep();
        }
        associate_ctrl_thread_mutex.unlock();
    }


    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::arm_disarm_vehicle(const bool& arm) {
        if (arm) {
            ROS_INFO("vehicle will be armed!");
            if (mavros_wrapper->set_arm_and_offboard()) {
                ROS_INFO("done!");
                logger->logger_on();
            }
        } else {
            ROS_INFO("vehicle will be disarmed!");
            if (mavros_wrapper->set_disarm()) {
                logger->logger_off();
                ROS_INFO("done!");
            }
        }
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    void TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::set_hover_pos(const Eigen::Vector3f& pos, const float& yaw) {
        mppi->setCostParams(pos, yaw);
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    bool TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::arm_disarm_srv_handle(tube_ctrl_srvs::SetArm::Request& req,
                                                                                            tube_ctrl_srvs::SetArm::Response& res) {
        bool arm_req = req.armed;
        arm_disarm_vehicle(arm_req);
        res.res = true;
        return true;
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    bool TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::hover_pos_srv_handle(tube_ctrl_srvs::SetHover::Request& req,
                                                                                           tube_ctrl_srvs::SetHover::Response& res) {
        Eigen::Vector3f pos_d(req.x_ned, req.y_ned, req.z_ned);
        float yaw_d = req.yaw;
        set_hover_pos(pos_d, yaw_d);
        res.res = true;
        return true;
    }

    template <int ROLLOUTS, int BLOCKSIZE_X, int NMPC_FREQ, int CTRL_FREQ>
    bool TubeController<ROLLOUTS, BLOCKSIZE_X, NMPC_FREQ, CTRL_FREQ>::takeoff_land_srv_handle(tube_ctrl_srvs::SetTakeoffLand::Request& req,
                                                                                              tube_ctrl_srvs::SetTakeoffLand::Response& res) {
        Eigen::Vector3f pos_d;
        fullstate_t _state = state_interface->get_state();
        if (ros::Time::now() - _state.timestamp >= ros::Duration(0.5)) {
            res.res = false;
            return false;
        }
        Eigen::Vector3f euler;
        get_euler_from_R<float>(euler, _state.R);
        float yaw_d = euler(2);
        if (req.takeoff) {
            std::cout << "[tube ctrl]: takeoff process yaw: " << yaw_d << std::endl;
            pos_d << _state.pos(0), _state.pos(1), -req.takeoff_altitude;
            std::cout << "[tube ctrl]: takeoff process start" << std::endl;
            arm_disarm_vehicle(true);
            std::cout << "[tube ctrl]: takeoff altitude: " << pos_d(2) << "m" << std::endl;
            set_hover_pos(pos_d, yaw_d);
        } else {
            std::cout << "[tube ctrl]: land process yaw: " << yaw_d << std::endl;
            pos_d << _state.pos(0), _state.pos(1), 0.1;
            set_hover_pos(pos_d, yaw_d);
            ros::Rate land_rate(20);
            while(ros::ok()) {
                _state = state_interface->get_state();
                if (_state.pos(2) > -0.1 && fabsf(_state.vel(2)) < 0.5f) {
                    ROS_INFO("detect landed: disarm");
                    arm_disarm_vehicle(false);
                    break;
                }
                land_rate.sleep();
            }
        }
        res.res = true;
        return true;
    }


}
