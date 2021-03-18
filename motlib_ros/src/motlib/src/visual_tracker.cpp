#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/features2d.hpp"

static const std::string OPENCV_WINDOW = "Image window";

auto makeHeader(const std::string& frame_id) -> std_msgs::Header {
    std_msgs::Header header;
    header.frame_id = frame_id;
    header.stamp = ros::Time::now();

    return header;
}

class ImageConverter {


public:
    ImageConverter() : m_image_transport{m_node_handle} {
        // Subscrive to input video feed and publish output video feed
        m_image_sub = m_image_transport.subscribe(
                "/image_publisher/output", 1, &ImageConverter::imageCb, this);
        m_image_pub = m_image_transport.advertise("/visual_tracker/output", 1);
    }

    ~ImageConverter() { cv::destroyWindow(OPENCV_WINDOW); }

    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(
                    msg, sensor_msgs::image_encodings::TYPE_8UC1);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        auto orb_detector = cv::ORB::create();
        std::vector<cv::KeyPoint> keypoints;
        orb_detector->detect(cv_ptr->image, keypoints);
        //-- Draw keypoints
        cv::Mat img_keypoints;
        drawKeypoints(cv_ptr->image, keypoints, img_keypoints);

        cv_bridge::CvImage cv_image(makeHeader("map"),
                                    sensor_msgs::image_encodings::TYPE_8UC3,
                                    img_keypoints);

        // Output modified video stream
        m_image_pub.publish(cv_image.toImageMsg());
    }

private:
    ros::NodeHandle m_node_handle;
    image_transport::ImageTransport m_image_transport;
    image_transport::Subscriber m_image_sub;
    image_transport::Publisher m_image_pub;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_converter");
    ImageConverter ic;
    ros::spin();
    return 0;
}