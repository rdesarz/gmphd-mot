#include "geometry_msgs/PoseArray.h"
#include "motlib/dynamic_models.hpp"
#include "motlib/filter.hpp"
#include "motlib/intensity.hpp"
#include "motlib/measurement_models.hpp"
#include "motlib/birth_models.hpp"
#include "motlib/timer.hpp"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud.h"


motlib::aligned_vec_t<typename motlib::TwoDimensionalLinearMeasurementModel<
        double>::MeasurementType>
        measurements;
void pointCloudCallback(const sensor_msgs::PointCloud& msg) {
    measurements.resize(msg.points.size());
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        measurements[i][0] = msg.points[i].x;
        measurements[i][1] = msg.points[i].y;
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_tracker");

    ros::NodeHandle n;

    ros::Subscriber measurements_sub =
            n.subscribe("measurements", 1, pointCloudCallback);

    ros::Publisher target_pub =
            n.advertise<geometry_msgs::PoseArray>("tracked_objects", 1);

    // Parameters
    double prob_survival = 0.99;
    double prob_detection = 0.99;
    double pruning_threshold = 1e-5;
    double merge_threshold = 5;
    double measurement_noise_cov = 0.2;
    double process_noise_cov = 0.1;
    double extraction_threshold = 0.5;

    // Configure components required for filter
    motlib::Timer timer;
    motlib::TwoDimensionalCVDynamicModel dynamic_model(timer,
                                                       process_noise_cov);
    motlib::RectangularFovSideAppearanceBirthModel birth_model(-5., 5., 0., 10., 0.2);
    motlib::TwoDimensionalLinearMeasurementModel measurement_model(
            measurement_noise_cov);

    // Instantiate filter
    motlib::GMPhdFilter filter{
            &dynamic_model, &measurement_model, &birth_model,   prob_survival,
            prob_detection, pruning_threshold,  merge_threshold};

    ros::Rate loop_rate(10);

    while (ros::ok()) {

        ros::spinOnce();

        timer.tick();
        filter.predict();
        filter.update(measurements);

        geometry_msgs::PoseArray tracked_objects;
        tracked_objects.header.frame_id = "sensor_frame";
        tracked_objects.header.stamp = ros::Time::now();

        for (const auto& component : filter.currentIntensity().components) {
            if (component.weight() > extraction_threshold) {
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
