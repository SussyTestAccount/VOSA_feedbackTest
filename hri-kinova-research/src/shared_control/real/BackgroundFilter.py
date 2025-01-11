#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
np.float = np.float64  # temp fix for following import, revert when ros_numpy repo is updated
import ros_numpy as rnp


def filter_background(cloud_msg):
    """
    Filters out points that are close to (0, 0, 0) and have low RGB values.
    """
    # Convert PointCloud2 to NumPy array
    cloud_array = rnp.point_cloud2.pointcloud2_to_array(cloud_msg)

    # Split the 'rgb' field to 'r', 'g', 'b'
    cloud_array = rnp.point_cloud2.split_rgb_field(cloud_array)

    # Filter the points based on RGB values and distance from origin
    # threshold = 64
    threshold = 180
    # threshold = 0
    mask = ((cloud_array['r'] >= threshold) | (cloud_array['g'] >= threshold) | (cloud_array['b'] >= threshold)) & \
           (np.sqrt(cloud_array['x'] ** 2 + cloud_array['y'] ** 2 + cloud_array['z'] ** 2) > 0.1)

    filtered_array = cloud_array[mask]

    # Merge 'r', 'g', 'b' fields back into 'rgb'
    filtered_array = rnp.point_cloud2.merge_rgb_fields(filtered_array)

    # Convert the filtered NumPy array back to PointCloud2
    filtered_cloud_msg = rnp.point_cloud2.array_to_pointcloud2(filtered_array, stamp=rospy.Time.now(),
                                                               # frame_id=cloud_msg.header.frame_id
                                                               frame_id="custom_camera_mount"
                                                               )
    return filtered_cloud_msg


class BackgroundFilter:
    def __init__(self):
        self.subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.point_cloud_callback)
        self.publisher = rospy.Publisher('/filtered_points', PointCloud2, queue_size=10)

    def point_cloud_callback(self, cloud_msg):
        """
        Callback Function for point cloud messages.
        """
        filtered_cloud = filter_background(cloud_msg)
        self.publisher.publish(filtered_cloud)


if __name__ == '__main__':
    rospy.init_node('point_cloud_filter')
    bg_filter = BackgroundFilter()
    rospy.spin()
