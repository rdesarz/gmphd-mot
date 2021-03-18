#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/highgui/highgui.hpp>

auto makeHeader(const std::string& frame_id) -> std_msgs::Header {
    std_msgs::Header header;
    header.frame_id = frame_id;
    header.stamp = ros::Time::now();

    return header;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle n;

    image_transport::ImageTransport image_transport(n);
    image_transport::Publisher image_pub =
            image_transport.advertise("/image_publisher/output", 1);


    ros::Rate loop_rate(10);

    auto image = cv::imread(
            "/home/hioot/Projects/motlib/motlib_ros/src/motlib/src/image.jpg",
            cv::IMREAD_GRAYSCALE);

    while (ros::ok()) {
        ros::spinOnce();

        cv_bridge::CvImage cv_image(makeHeader("map"),
                                    sensor_msgs::image_encodings::TYPE_8UC1,
                                    image);

        // Output modified video stream
        image_pub.publish(cv_image.toImageMsg());

        loop_rate.sleep();
    }

    return 0;
}