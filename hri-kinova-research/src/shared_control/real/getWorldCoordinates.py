#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import PointCloud
from AssistiveTeleoperation import AssistiveTeleoperation
from ForwardKinematics import ForwardKinematicsKinova
from ActuatorModel import ActuatorModel
from kortex_driver.msg import BaseCyclic_Feedback
import matplotlib.pyplot as plt
from std_msgs.msg import Header
from geometry_msgs.msg import Point32


class GetWorldCoords:
    def __init__(self):
        print("GET WORLD COORDINATES")
        self.curr_pos = []
        self.positions = []
        rospy.init_node('get_world_coords')
        rospy.Subscriber("/clusters", PointCloud, self.centroid_callback)
        self.fk_kinova = ForwardKinematicsKinova()
        self.pub = rospy.Publisher('/world_cords', PointCloud, queue_size=10)
        self.basefeedback = rospy.Subscriber("/" + 'my_gen3' + "/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback, buff_size=1)
        rospy.Timer(rospy.Duration(1), self.update_joints)

    def base_feedback_callback(self, feedback):
        get_state = ActuatorModel()
        get_state.position_0 = feedback.actuators[0].position
        get_state.position_1 = feedback.actuators[1].position
        get_state.position_2 = feedback.actuators[2].position
        get_state.position_3 = feedback.actuators[3].position
        get_state.position_4 = feedback.actuators[4].position
        get_state.position_5 = feedback.actuators[5].position
        get_state.position_6 = feedback.actuators[6].position

        self.positions.append(get_state)
        self.curr_pos = get_state.get_position()

    def get_curr_pos(self):
        if len(self.positions) == 0:
            return 0

        actuatorModel = ActuatorModel()
        curr_pos = self.positions.pop(len(self.positions)-1)
        joint_angles = actuatorModel.setActuatorData(curr_pos.position_0, curr_pos.position_1,
                                            curr_pos.position_2, curr_pos.position_3,
                                            curr_pos.position_4, curr_pos.position_5,
                                            curr_pos.position_6)

        return joint_angles

    def centroid_callback(self, data):
        print(data)
        self.list_centroids = list()
        for point in data.points:
            self.list_centroids.append([point.x, point.y, point.z])

        # Cast to NumpyArray
        self.list_centroids = np.array(self.list_centroids)

        # Camera to World Conversion
        world_coordindates = self.fk_kinova.camera_xyz_to_world(self.list_centroids.T).T
        print(world_coordindates)
        self.publish_world_coordinates(world_coordindates)

    # def centroid_callback(self, data):
    #     point_data = data.point
    #     # print(point_data)
    #     point_matrix = np.array([[point_data.x], [point_data.y], [point_data.z], [1]])
    #     # print("Point Matrix", point_matrix)
    #     world_coordinates = self.fk_kinova.camera_xyz_to_world(point_matrix)
    #     print(world_coordinates)
    #     self.publish_world_coordinates(world_coordinates, data.header)

    def publish_world_coordinates(self, coords):
        centroid_cloud = PointCloud()
        centroid_cloud.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
        centroid_cloud.points = [Point32(x=c[0], y=c[1], z=c[2]) for c in coords]
        self.pub.publish(centroid_cloud)

    def update_joints(self, event):
        joint_angles_deg = self.get_curr_pos()
        joint_angles = np.radians(joint_angles_deg)
        # print("Joint Angles ", joint_angles)
        self.fk_kinova.update_joints(joint_angles)

    def plot_alpha_values(self, ax):
        plt.cla()
        self.fk_kinova.draw(ax=ax)
        plt.pause(0.1)

if __name__ == '__main__':
    node = GetWorldCoords()
    rate = rospy.Rate(10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    try:
        while not rospy.is_shutdown():
            # node.plot_alpha_values(ax)
            # print(node.fk_kinova.fk())
            rate.sleep()
    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)

