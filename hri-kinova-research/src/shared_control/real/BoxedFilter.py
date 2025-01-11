#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
np.float = np.float64  # temp fix for following import, revert when ros_numpy repo is updated
import ros_numpy as rnp
from kortex_driver.msg import BaseCyclic_Feedback
from ActuatorModel import ActuatorModel
from ForwardKinematics import ForwardKinematicsKinova
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud


def filter_background(cloud_msg, fk):
    """
    Filters out points that are close to (0, 0, 0) and have low RGB values.
    """

    X_LIMS = [0.0, 1.17]
    Y_LIMS = [0.20, 0.70]
    Z_LIMS = [-0.005, 10.0]

    # Convert PointCloud2 to NumPy array
    cloud_array = rnp.point_cloud2.pointcloud2_to_array(cloud_msg)

    # Split the 'rgb' field to 'r', 'g', 'b'
    cloud_array = rnp.point_cloud2.split_rgb_field(cloud_array)

    # Filter the points based on RGB values and distance from origin
    # threshold = 64
    threshold = 180
    mask = ((cloud_array['r'] >= threshold) | (cloud_array['g'] >= threshold) | (cloud_array['b'] >= threshold)) & \
           (np.sqrt(cloud_array['x'] ** 2 + cloud_array['y'] ** 2 + cloud_array['z'] ** 2) > 0.1)

    filtered_array = cloud_array[mask]

    # Convert NP Tuple Array to Matrix for transformation
    xyz = np.vstack((filtered_array['x'], filtered_array['y'], filtered_array['z']))
    world_coords = fk.camera_xyz_to_world(xyz)
    # print(world_coords)

    # Replace x, y, z values in filtered_array
    filtered_array["x"], filtered_array["y"], filtered_array["z"] = world_coords[0], world_coords[1], world_coords[2]

    # print(len(filtered_array))
    # This complex block of garbage filters the point cloud into the prism formed by {X,Y,Z}_LIMS
    filter_mask = np.where(
        (X_LIMS[0] < filtered_array["x"]) &
        (filtered_array["x"] < X_LIMS[1]) &
        (Y_LIMS[0] < filtered_array["y"]) &
        (filtered_array["y"] < Y_LIMS[1]) &
        (Z_LIMS[0] < filtered_array["z"]) &
        (filtered_array["z"] < Z_LIMS[1])
    )
    filtered_array = filtered_array[filter_mask]
    # print(len(filtered_array))

    # Merge 'r', 'g', 'b' fields back into 'rgb'
    filtered_array = rnp.point_cloud2.merge_rgb_fields(filtered_array)

    # Convert the filtered NumPy array back to PointCloud2
    filtered_cloud_msg = rnp.point_cloud2.array_to_pointcloud2(filtered_array, stamp=rospy.Time.now(),
                                                               # frame_id=cloud_msg.header.frame_id
                                                               frame_id="base_link"
                                                               )
    return filtered_cloud_msg


class BackgroundFilter:
    def __init__(self):
        self.curr_pos = None
        self.subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.point_cloud_callback)
        self.publisher = rospy.Publisher('/filtered_boxed_point_cloud', PointCloud2, queue_size=10)
        self.basefeedback = rospy.Subscriber("/" + 'my_gen3' + "/base_feedback", BaseCyclic_Feedback,
                                             self.base_feedback_callback, buff_size=1)
        self.positions = []
        self.fk_kinova = ForwardKinematicsKinova()
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

    def point_cloud_callback(self, cloud_msg):
        """
        Callback Function for point cloud messages.
        """
        filtered_cloud = filter_background(cloud_msg, self.fk_kinova)
        self.publisher.publish(filtered_cloud)

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

    def publish_world_coordinates(self, coords):
        centroid_cloud = PointCloud()
        centroid_cloud.header = Header(stamp=rospy.Time.now(), frame_id="base_link")
        centroid_cloud.points = [Point32(x=c[0], y=c[1], z=c[2]) for c in coords]
        self.publisher.publish(centroid_cloud)

    def update_joints(self, event):
        joint_angles_deg = self.get_curr_pos()
        joint_angles = np.radians(joint_angles_deg)
        # print("Joint Angles ", joint_angles)
        self.fk_kinova.update_joints(joint_angles)


if __name__ == '__main__':
    rospy.init_node('box_cloud_filter')
    bg_filter = BackgroundFilter()
    rospy.spin()
