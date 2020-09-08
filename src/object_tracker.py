#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point32, PoseArray, Pose
from sensor_msgs.msg import PointCloud
import copy
import numpy as np
import gmphd.birth_models, gmphd.clutter_models, gmphd.dynamic_models, gmphd.process_noises 
from gmphd.prediction import predict_intensity
from gmphd.update import update_intensity
from gmphd.postprocessing import apply_pruning, apply_merging

# Callback to get the most recent measurement
measurements = list()
def measurements_cb(data):
    global measurements
    measurements = [np.array([point.x, point.y]) for point in data.points]
        
def make_pose_from_target(target):
    pose = Pose()
    pose.position.x = target.mean[0]
    pose.position.y = target.mean[1]

    return pose

# Main loop
def track_object():
    rospy.init_node('object_tracker', anonymous=True)

    # I/O
    tracked_objects_pub = rospy.Publisher(
        'tracked_objects', PoseArray, queue_size=1)

    rospy.Subscriber("measurements", PointCloud, measurements_cb)
    rate = rospy.Rate(10)

    # Tracker parameters
    width = rospy.get_param("~width", 5.)
    depth = rospy.get_param("~depth", 5.)
    birth_model = gmphd.birth_models.SquaredFieldOfView(width, depth, 2.)

    prob_survival = 0.99
    process_noise_sd = 0.2
    dynamic_model = gmphd.dynamic_models.ConstantVelocity()
    process_noise = gmphd.process_noises.ConstantVelocity(process_noise_sd)

    prob_detection = 0.99
    measurement_model = np.array([[1., 0., 0., 0.],
                                  [0., 1., 0., 0.]])
    measurement_noise = np.array([[0.1, 0.],
                                  [0., 0.1]])
    clutter_model = gmphd.clutter_models.Constant(0.)

    pruning_threshold = 1e-5
    merging_threshold = 4
    extraction_threshold = 0.5

    intensity = list()
    last_step_time = rospy.Time.now()
    while not rospy.is_shutdown():
        # Compute elapsed time since last update
        now = rospy.Time.now()
        delta_t = (now-last_step_time).to_sec()
        last_step_time = now

        # Filter loop
        intensity = predict_intensity(intensity, dynamic_model(delta_t), process_noise(delta_t), prob_survival, birth_model)
        intensity = update_intensity(intensity, measurements, measurement_model, measurement_noise, prob_detection, clutter_model)
        intensity = apply_pruning(intensity, pruning_threshold)
        intensity = apply_merging(intensity, merging_threshold)

        # Publish extracted targets
        tracked_objects = PoseArray()
        tracked_objects.header.frame_id = "sensor_frame"
        tracked_objects.header.stamp = now
        tracked_objects.poses = [make_pose_from_target(target) for target in intensity if target.weight > extraction_threshold]
        tracked_objects_pub.publish(tracked_objects)

        rate.sleep()


if __name__ == '__main__':
    try:
        track_object()
    except rospy.ROSInterruptException:
        pass
