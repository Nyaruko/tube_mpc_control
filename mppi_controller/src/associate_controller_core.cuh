#ifndef ASC_CORE_CUH
#define ASC_CORE_CUH

#include <atomic>
#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include "math_static_use.h"

namespace mppi_control {
class AssociateCtrlCore {
public:
        typedef struct state_s{
            Eigen::Vector3f pos;
            Eigen::Vector3f vel;
            Eigen::Vector3f acc;
            state_s() {
                pos.setZero();
                vel.setZero();
                acc.setZero();
            }
        } state_t;

        typedef struct u_s{
            Eigen::Vector3f acc_d;
            float yaw_d;
            u_s(){
                acc_d.setZero();
                yaw_d = 0;
            }
        } u_t;

        typedef struct param_s{
            Eigen::Vector3f k_p_xy;
            Eigen::Vector3f k_p_z;
            Eigen::Vector3f k_d_xy;
            Eigen::Vector3f k_d_z;
            Eigen::Vector3f k_i_xy;
            Eigen::Vector3f k_i_z;
            float k_ff_xy;
            float k_ff_z;
            float sat_d_xy;
            float sat_d_z;
            float sat_i_xy;
            float sat_i_z;
            param_s() {
                k_p_xy.setZero();
                k_p_z.setZero();
                k_d_xy.setZero();
                k_d_z.setZero();
                k_i_xy.setZero();
                k_i_z.setZero();
                k_ff_xy = 0;
                k_ff_z = 0;
                sat_d_xy = 0;
                sat_d_z = 0;
                sat_i_xy = 0;
                sat_i_z = 0;
            }
        } param_t;

        typedef Eigen::Vector3f ctrl_gain_t;

        AssociateCtrlCore(ros::NodeHandle node, float ctrl_freq):  ctrl_freq_(ctrl_freq){
            Eigen::Vector3f _kp_xy = {1.2, 2.4, 0.02};
            Eigen::Vector3f _kp_z = {1.2, 2.7, 0.02};
            Eigen::Vector3f _kd_xy = {0.002, 0.003, 0.0003};
            Eigen::Vector3f _kd_z = {0.002, 0.004, 0.0005};
            Eigen::Vector3f _ki_xy = {0.15, 0.00, 0.00};
            Eigen::Vector3f _ki_z = {0.15, 0.00, 0.00};
            float _kff_xy = 0.1;
            float _kff_z = 0.1;
            float _sat_d_xy = 2.0;
            float _sat_d_z = 2.0;
            float _sat_i_xy = 8.0;
            float _sat_i_z = 15.0;
            node.param<float>("Kp_xy/P", _kp_xy[0], 1.2);
            std::cout << _kp_xy.transpose() << std::endl;
            node.param<float>("Kp_xy/V", _kp_xy[1], 2.4);
            node.param<float>("Kp_xy/A", _kp_xy[2], 0.02);
            node.param<float>("Kp_z/P", _kp_z[0], 1.2);
            node.param<float>("Kp_z/V", _kp_z[1], 2.7);
            node.param<float>("Kp_z/A", _kp_z[2], 0.02);
            node.param<float>("Kd_xy/P", _kd_xy[0], 0.002);
            node.param<float>("Kd_xy/V", _kd_xy[1], 0.003);
            node.param<float>("Kd_xy/A", _kd_xy[2], 0.0003);
            node.param<float>("Kd_z/P", _kd_z[0], 0.002);
            node.param<float>("Kd_z/V", _kd_z[1], 0.004);
            node.param<float>("Kd_z/A", _kd_z[2], 0.0005);
            node.param<float>("Ki_xy/P", _ki_xy[0], 0.15);
            node.param<float>("Ki_xy/V", _ki_xy[1], 0.0);
            node.param<float>("Ki_xy/A", _ki_xy[2], 0.0);
            node.param<float>("Ki_z/P", _ki_z[0], 0.15);
            node.param<float>("Ki_z/V", _ki_z[1], 0.00);
            node.param<float>("Ki_z/A", _ki_z[2], 0.00);
            node.param<float>("Kff_xy", _kff_xy, 0.1);
            node.param<float>("Kff_z", _kff_z, 0.1);
            node.param<float>("Sat_d_xy", _sat_d_xy, 2.0);
            node.param<float>("Sat_d_z", _sat_d_z, 2.0);
            node.param<float>("Sat_i_xy", _sat_d_xy, 8.0);
            node.param<float>("Sat_i_z", _sat_d_z, 15.0);
            ctrl_param_.k_p_xy  = _kp_xy;
            ctrl_param_.k_p_z = _kp_z;
            ctrl_param_.k_d_xy = _kd_xy;
            ctrl_param_.k_d_z = _kd_z;
            ctrl_param_.k_i_xy = _ki_xy;
            ctrl_param_.k_i_z = _ki_z;
            ctrl_param_.k_ff_xy = _kff_xy;
            ctrl_param_.k_ff_z = _kff_z;
            ctrl_param_.sat_d_xy = _sat_d_xy;
            ctrl_param_.sat_d_z = _sat_d_z;
            ctrl_param_.sat_i_xy = _sat_i_xy;
            ctrl_param_.sat_i_z = _sat_i_z;
            K_p_xy_ << ctrl_param_.k_p_xy;
            K_p_z_ << ctrl_param_.k_p_z;
            K_d_xy_ << ctrl_param_.k_d_xy;
            K_d_z_ << ctrl_param_.k_d_z;
            K_i_xy_ << ctrl_param_.k_i_xy;
            K_i_z_ << ctrl_param_.k_i_z;
            K_ff_.diagonal() << ctrl_param_.k_ff_xy, ctrl_param_.k_ff_xy, ctrl_param_.k_ff_z;
            last_e_x_.setZero();
            int_e_x_.setZero();
            last_e_y_.setZero();
            int_e_y_.setZero();
            last_e_z_.setZero();
            int_e_z_.setZero();
        }

        ~AssociateCtrlCore() {}

        void reset_ctrl() {
            last_e_x_.setZero();
            int_e_x_.setZero();
            last_e_y_.setZero();
            int_e_y_.setZero();
            last_e_z_.setZero();
            int_e_z_.setZero();
        }

        void cal_u(float* s, float* s_d, float* u_d, Eigen::Vector3f& u) {
            Eigen::Vector3f e_p;
            e_p << s_d[0]-s[0], s_d[1]-s[1], s_d[2]-s[2];
            Eigen::Vector3f e_v;
            e_v << s_d[3]-s[3], s_d[4]-s[4], s_d[5]-s[5];
            Eigen::Vector3f e_a;
            e_a << s_d[6]-s[6], s_d[7]-s[7], s_d[8]-s[8];

            Eigen::Vector3f u_nominal;
            u_nominal << s_d[6], s_d[7], s_d[8];//u_d[0], u_d[1], u_d[2];
//            u_nominal << 0,0,0;
//            std::cout << "u_acc: " << u_nominal.transpose() << std::endl;
//            std::cout << e_p.transpose() << ", " << e_v.transpose() << ", " << e_a.transpose() << std::endl;

            Eigen::Vector3f e_x_;
            e_x_<< e_p(0), e_v(0), e_a(0);
            Eigen::Vector3f e_y_;
            e_y_<< e_p(1), e_v(1), e_a(1);
            Eigen::Vector3f e_z_;
            e_z_<< e_p(2), e_v(2), e_a(2);

            Eigen::Vector3f d_e_x_ = (e_x_ - last_e_x_)*ctrl_freq_;
            Eigen::Vector3f d_e_y_ = (e_y_ - last_e_y_)*ctrl_freq_;
            Eigen::Vector3f d_e_z_ = (e_z_ - last_e_z_)*ctrl_freq_;
            sat_check(d_e_x_, 1);
            sat_check(d_e_y_, 1);
            sat_check(d_e_z_, 1);

            int_e_x_ += e_x_/ctrl_freq_;
            int_e_y_ += e_y_/ctrl_freq_;
            int_e_z_ += e_z_/ctrl_freq_;
            sat_check(int_e_x_, 2);
            sat_check(int_e_y_, 2);
            sat_check(int_e_z_, 2);

            Eigen::Vector3f u_ac;
            u_ac<< e_x_.transpose()*K_p_xy_
                    + d_e_x_.transpose()*K_d_xy_
                    + int_e_x_.transpose()*K_i_xy_,
                    e_y_.transpose()*K_p_xy_
                    + d_e_y_.transpose()*K_d_xy_
                    + int_e_y_.transpose()*K_i_xy_,
                    e_z_.transpose()*K_p_z_
                    + d_e_z_.transpose()*K_d_z_
                    + int_e_z_.transpose()*K_i_z_;
            Eigen::Vector3f _G_vector(0,0,-ONE_G);
            u_ac = u_ac + _G_vector;

            Eigen::Vector3f u_ff;
            u_ff << u_d[0], u_d[1], u_d[2];
            u = u_nominal + u_ac + K_ff_*u_ff;
        }

        inline float signmun(float _i) {
            return (float)((_i>0.0)-(_i<0.0));
        }

        bool sat_check(Eigen::Vector3f& _vect, int _order) {
            bool _sat_flag = false;
            for (int i = 0; i < 3; i++) {
                if (_order == 1) {
                    if (i < 2) {
                        bool tmp_flag= (fabsf(_vect(i)) > ctrl_param_.sat_d_xy);
                        _vect(i) = (tmp_flag) ? (signmun(_vect(i))*ctrl_param_.sat_d_xy) : _vect(i);
                        _sat_flag = (tmp_flag) ? true: _sat_flag;
                    } else {
                        bool tmp_flag= (fabsf(_vect(i)) > ctrl_param_.sat_d_z);
                        _vect(i) = (tmp_flag) ? (signmun(_vect(i))*ctrl_param_.sat_d_z) : _vect(i);
                        _sat_flag = (tmp_flag) ? true: _sat_flag;
                    }
                } else if (_order == 2) {
                    if (i < 2) {
                        bool tmp_flag= (fabsf(_vect(i)) > ctrl_param_.sat_i_xy);
                        _vect(i) = (tmp_flag) ? (signmun(_vect(i))*ctrl_param_.sat_i_xy) : _vect(i);
                        _sat_flag = (tmp_flag) ? true: _sat_flag;
                    } else {
                        bool tmp_flag= (fabsf(_vect(i)) > ctrl_param_.sat_i_z);
                        _vect(i) = (tmp_flag) ? (signmun(_vect(i))*ctrl_param_.sat_i_z) : _vect(i);
                        _sat_flag = (tmp_flag) ? true: _sat_flag;
                    }
                }
            }
            return _sat_flag;
        }

    private:
        param_t ctrl_param_;
        ctrl_gain_t K_p_xy_;
        ctrl_gain_t K_p_z_;
        ctrl_gain_t K_i_xy_;
        ctrl_gain_t K_i_z_;
        ctrl_gain_t K_d_xy_;
        ctrl_gain_t K_d_z_;
        Eigen::Matrix3f K_ff_;
        Eigen::Vector3f last_e_x_;
        Eigen::Vector3f int_e_x_;
        Eigen::Vector3f last_e_y_;
        Eigen::Vector3f int_e_y_;
        Eigen::Vector3f last_e_z_;
        Eigen::Vector3f int_e_z_;

        float ctrl_freq_;
    };
}
#endif