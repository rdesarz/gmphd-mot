#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sstream>

#include "motlib/model.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_node");

  ros::NodeHandle n;

  ros::Rate loop_rate(10);

  while (ros::ok()) {

    ROS_INFO("%d", computeAddition());

    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}
