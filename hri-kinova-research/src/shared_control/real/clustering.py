#!/usr/bin/env python3

import rospy
import threading
import numpy as np
from sensor_msgs.msg import PointCloud2, PointCloud
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Int32
from sklearn.cluster import KMeans
from geometry_msgs.msg import Point32

latest_point_cloud = None
n_clusters = 1  # Default value, will be updated by the subscriber
cluster_lock = threading.Lock()  # Lock for thread-safe access to n_clusters


def num_objects_callback(data):
    global n_clusters
    with cluster_lock:
        n_clusters = data.data


def down_sample_points(points, voxel_size=0.05):
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
    idx_pts_vox_sorted = np.argsort(inverse)

    voxel_grid = {}
    grid_candidate_center = []
    last_seen = 0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
        grid_candidate_center.append(np.mean(voxel_grid[tuple(vox)], axis=0))
        last_seen += nb_pts_per_voxel[idx]

    return np.array(grid_candidate_center)


def point_cloud_callback(data):
    global latest_point_cloud
    latest_point_cloud = data


def process_and_publish_centroids(event, centroid_pub):
    global latest_point_cloud
    if latest_point_cloud is not None:
        points = np.array(list(pc2.read_points(latest_point_cloud, skip_nans=True, field_names=("x", "y", "z"))))
        if points.size != 0:
            down_sampled_points = down_sample_points(points)

            # Only cluster if number of points is less than k.
            if len(down_sampled_points) < n_clusters:
                return

            with cluster_lock:  # Ensure thread safe read
                current_n_clusters = max(1, n_clusters)  # At least one cluster

            kmeans = KMeans(n_clusters=current_n_clusters, random_state=0, n_init=5, max_iter=200)  # Finetune n_init and max_iter
            kmeans.fit(down_sampled_points)
            centroids = kmeans.cluster_centers_
        else:
            centroids = np.array([])

        centroid_cloud = PointCloud()
        centroid_cloud.header = Header(stamp=rospy.Time.now(), frame_id=latest_point_cloud.header.frame_id)
        centroid_cloud.points = [Point32(x=c[0], y=c[1], z=c[2]) for c in centroids]
        for c in centroids:
            print("CENTROIDS: ", c[0], c[1], c[2])

        centroid_pub.publish(centroid_cloud)


def process_point_cloud_thread(data, centroid_pub):
    thread = threading.Thread(target=process_and_publish_centroids, args=(data, centroid_pub))
    thread.start()


def point_cloud_listener():
    rospy.init_node('point_cloud_processor', anonymous=True)
    centroid_pub = rospy.Publisher('/clusters', PointCloud, queue_size=10)
    rospy.Subscriber("/filtered_boxed_point_cloud", PointCloud2, point_cloud_callback)
    rospy.Subscriber("/num_objects", Int32, num_objects_callback)

    timer_duration = 0.5  # Adjust the duration for faster updates
    rospy.Timer(rospy.Duration(timer_duration), lambda event: process_and_publish_centroids(event, centroid_pub))

    rospy.spin()


if __name__ == '__main__':
    point_cloud_listener()

