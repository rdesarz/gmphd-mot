#include "motlib/filter.hpp"
#include "motlib/timer.hpp"
#include "motlib/intensity.hpp"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud.h"
#include "geometry_msgs/PoseArray.h"

namespace tracker2d {

    struct DynamicModel {
        DynamicModel(const motlib::Timer& timer, double process_noise) : m_timer{timer}, m_process_noise_std_dev{process_noise}
        {
            // clang-format off
            const auto delta_t = m_timer.getElapsedTime();
            m_state_transition_matrix << 1, 0, delta_t,       0,
                                         0, 1,       0, delta_t,
                                         0, 0,       1,       0,
                                         0, 0,       0,       1;

            m_process_noise <<
            std::pow(delta_t, 4)/4.,                     0., std::pow(delta_t, 3)/2,                        0,
                                 0., std::pow(delta_t, 4)/4,                     0., std::pow(delta_t, 3) / 2,
            std::pow(delta_t, 3)/2.,                     0.,   std::pow(delta_t, 2),                       0.,
                                 0., std::pow(delta_t, 3)/2,                     0.,     std::pow(delta_t, 2);
            // clang-format on
        }

        const Eigen::Matrix4d& state_transition() {
            m_state_transition_matrix(0,2) = m_timer.getElapsedTime();
            m_state_transition_matrix(1,3) = m_timer.getElapsedTime();
            return m_state_transition_matrix;
        }

        const Eigen::Matrix4d& process_noise() { 
            const auto delta_t = m_timer.getElapsedTime();
            const double squared_noise_std = std::pow(m_process_noise_std_dev, 2);
            
            m_process_noise(0,0) = std::pow(delta_t, 4)/4*squared_noise_std;
            m_process_noise(0,2) = std::pow(delta_t, 3)/2*squared_noise_std;
            m_process_noise(1,1) = std::pow(delta_t, 4)/4*squared_noise_std;
            m_process_noise(1,3) = std::pow(delta_t, 3)/2*squared_noise_std;
            m_process_noise(2,0) = std::pow(delta_t, 3)/2*squared_noise_std;
            m_process_noise(2,2) = std::pow(delta_t, 2)*squared_noise_std;
            m_process_noise(3,1) = std::pow(delta_t, 3)/2*squared_noise_std;
            m_process_noise(3,3) = std::pow(delta_t, 2)*squared_noise_std;

            return m_process_noise;
        }

    private:
        Eigen::Matrix4d m_state_transition_matrix;
        Eigen::Matrix4d m_process_noise;
        double m_process_noise_std_dev;
        const motlib::Timer &m_timer;
    };

    class MeasurementModel {
    public:
        MeasurementModel() {
            // clang-format off
            m_state_transition_matrix << 1, 0, 0, 0,
                                         0, 1, 0, 0;

            m_measurement_noise << 0.1,   0,
                                     0, 0.1;
            // clang-format on
        }

        const Eigen::Matrix<double, 2, 4>& state_transition() {
            return m_state_transition_matrix;
        }

        const Eigen::Matrix2d& measurement_noise() {
            return m_measurement_noise;
        }

    private:
        Eigen::Matrix<double, 2, 4> m_state_transition_matrix;
        Eigen::Matrix2d m_measurement_noise;
    };

    struct BirthModel {
        BirthModel() {
            // Create a squared field of view with possible appeareance on the side of it
            for(int i=0;i<10;++i)
            {
                // clang-format off
                m_intensity.components.push_back(motlib::GaussianMixtureComponent4d{});
                m_intensity.components.back().weight() = 0.1;
                m_intensity.components.back().mean() << -5, static_cast<double>(i), 0.2, 0.;
                m_intensity.components.back().covariance() << 0.5, 0, 0, 0,
                                                            0, 0.5, 0, 0,
                                                            0, 0, 0.1, 0,
                                                            0, 0, 0, 0.1;

                m_intensity.components.push_back(motlib::GaussianMixtureComponent4d{});
                m_intensity.components.back().weight() = 0.1;
                m_intensity.components.back().mean() << 5, static_cast<double>(i), -0.2, 0.;
                m_intensity.components.back().covariance() << 0.5, 0, 0, 0,
                                                            0, 0.5, 0, 0,
                                                            0, 0, 0.1, 0,
                                                            0, 0, 0, 0.1;
                // clang-format on
            }
        }

        const motlib::Intensity4d& intensity() const { return m_intensity; }

        motlib::Intensity4d m_intensity;
    };

};// namespace

motlib::aligned_vec_t<Eigen::Vector2d> measurements;
void pointCloudCallback(const sensor_msgs::PointCloud& msg)
{
    measurements.clear();
    for(const auto& point : msg.points)
    {
        measurements.push_back(Eigen::Vector2d{});
        measurements.back()[0] = point.x;
        measurements.back()[1] = point.y;
    }
}


int main(int argc, char **argv) {
  ros::init(argc, argv, "object_tracker");

  ros::NodeHandle n;

  ros::Subscriber measurements_sub = n.subscribe("measurements", 1, pointCloudCallback);

  ros::Publisher target_pub = n.advertise<geometry_msgs::PoseArray>("tracked_objects", 1);

  double prob_survival = 0.99;
  double prob_detection = 0.99;
  double pruning_threshold = 1e-5;
  double merge_threshold = 5;

  // Configure predict components
  motlib::Timer timer;
  tracker2d::DynamicModel dynamic_model(timer, 0.1);
  tracker2d::BirthModel birth_model;

  // Configure update components
  tracker2d::MeasurementModel measurement_model;

  // Instantiate filter
  motlib::Filter filter{&dynamic_model, &measurement_model, &birth_model, prob_survival, prob_detection, pruning_threshold, merge_threshold};

  ros::Rate loop_rate(10);

  while (ros::ok()) {

    ros::spinOnce();

    timer.tick();
 
    filter.predict();
 
    filter.update(measurements);

    geometry_msgs::PoseArray tracked_objects;
    tracked_objects.header.frame_id = "sensor_frame";
    tracked_objects.header.stamp = ros::Time::now();

    for(const auto& component : filter.currentIntensity().components)
    {
        if(component.weight() > 0.5)
        {
            geometry_msgs::Pose pose;
            pose.position.x = component.mean()[0];
            pose.position.y = component.mean()[1]; 
            tracked_objects.poses.push_back(pose);
        }
    }   

    target_pub.publish(tracked_objects);

    loop_rate.sleep();
  }

  return 0;
}
