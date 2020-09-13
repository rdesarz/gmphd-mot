#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import random
import numpy as np
import collections

Object = collections.namedtuple('Object', 'position velocity')

def compute_motion_matrix(delta_t):
    return np.array([[delta_t, 0.], [0., delta_t]])


def remove_out_of_fov_target(targets, x_min, x_max):
    targets = [target for target in targets if target.position.x <= x_max and target.position.x >= x_min]

    return targets


def add_new_target(targets, x_min, x_max, y_min, y_max, velocity):
    point = Point32()
    # There is even chance to pop from one side or the other
    if random.uniform(0, 1) > 0.5:
        point.x = x_max
        target = Object(position=point, velocity=-velocity)
    else:
        point.x = x_min
        target = Object(position=point, velocity=velocity)

    target.position.y = random.uniform(y_min, y_max)
    targets.append(target)


def generate_measurements():
    rospy.init_node('fake_point_target_generator', anonymous=True)

    measurement_pub = rospy.Publisher(
        'sensor_measurements', PointCloud, queue_size=1)
    ground_truth_pub = rospy.Publisher(
        'ground_truth', PointCloud, queue_size=1)

    rate = rospy.Rate(20)

    random.seed()

    # Shape of FOV (Square)
    x_min = rospy.get_param("~x_min", -5.)
    x_max = rospy.get_param("~x_max",  5.)
    y_min = rospy.get_param("~y_min", -5.)
    y_max = rospy.get_param("~y_max",  5.)

    # Maximum number of targets
    targets_max_nb = rospy.get_param("~targets_max_nb", 2.)

    # Measurement noise
    measurement_noise = rospy.get_param("~measurement_noise", 0.1)

    # Target dynamic
    target_velocity = rospy.get_param("~target_velocity", 1.)

    # Probability to pop a new target
    p_new_target = rospy.get_param("~p_new_target", 0.05)

    # Create a single target at the beginning
    targets = list()
    add_new_target(targets, x_min, x_max, y_min, y_max, target_velocity)

    last_step_time = rospy.Time.now()

    while not rospy.is_shutdown():
        # Compute elapsed time since last update
        now = rospy.Time.now()
        delta_t = (now-last_step_time).to_sec()
        last_step_time = now

        # Generate the ground truth of a moving target
        ground_truth = PointCloud()
        ground_truth.header.frame_id = "sensor_frame"
        ground_truth.header.stamp = now

        # Update all targets
        for target in targets:
            pos = np.array([target.position.x, target.position.y])
            speed = np.array([target.velocity, 0]).transpose()
            pos += compute_motion_matrix(delta_t).dot(speed)
            target.position.x = pos[0]
            target.position.y = pos[1]

        ground_truth.points = [target.position for target in targets]

        # Generate a fake measurement of the moving target with gaussian noise
        sensor_measurement = PointCloud()
        sensor_measurement.header.frame_id = "sensor_frame"
        sensor_measurement.header.stamp = now

        for gt_point in ground_truth.points:
            point = Point32()
            point.x = gt_point.x + random.gauss(0, measurement_noise)
            point.y = gt_point.y + random.gauss(0, measurement_noise)
            sensor_measurement.points.append(point)

        measurement_pub.publish(sensor_measurement)
        ground_truth_pub.publish(ground_truth)

        targets = remove_out_of_fov_target(targets, x_min, x_max)

        if random.uniform(0, 1) < p_new_target and len(targets) < targets_max_nb:
            add_new_target(targets, x_min, x_max, y_min,
                           y_max, target_velocity)

        rate.sleep()


if __name__ == '__main__':
    try:
        generate_measurements()
    except rospy.ROSInterruptException:
        pass
