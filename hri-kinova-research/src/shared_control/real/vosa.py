#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from sag import SAGTeleoperation
from constants import STOP_SCAN_THRESHOLD, PLACEMENT_THRESHOLDS, HOME
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
from custom_msgs.msg import CentroidConfidence

class VOSATeleoperation(SAGTeleoperation):
    def __init__(self):
        super().__init__()

        self.intermediate_position_reached = True
        self.pick_set = []
        self.adjusted_z = None
        self.Z_is_updated = False
        
        self.confidence_publisher = rospy.Publisher()
        self.centroid_conf_pub = rospy.Publisher('/goal_confidence_centroids', CentroidConfidenceArray, queue_size=10)
        
        rospy.Subscriber("/clusters", PointCloud, self.centroid_callback)
        rospy.loginfo("[VOSA] Initialized with dynamic pick set subscription.")

    def centroid_callback(self, msg):
        if self.ee_position[0] > STOP_SCAN_THRESHOLD:
            return

        detected = []
        for pt in msg.points:
            centroid = np.array([pt.x + 0.025, pt.y - 0.01, pt.z])
            if pt.y > PLACEMENT_THRESHOLDS[self.task]:
                detected.append(centroid)

        if not self.intermediate_position_reached and len(detected) == 0:
            return

        self.intermediate_position_reached = True

        if self.gripper_closed and not self.Z_is_updated:
            self.adjusted_z = self.ee_position[2]
            for i in range(len(detected)):
                detected[i][2] = 0.08 + self.adjusted_z
            self.Z_is_updated = True

        if not self.gripper_closed:
            self.Z_is_updated = False

        self.pick_set = detected
        rospy.loginfo(f"[VOSA] Updated pick set with {len(detected)} centroids.")

    def main(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.current_goal_set = self.place_set if self.gripper_closed else self.pick_set

            if self.input_magnitude(self.uh) > 1e-6:
                self.last_nonzero_uh = self.uh

            if self.Y_pressed:
                rospy.loginfo("Resetting to home...")
                self.twist_command(-0.45, 0, 0.45)
                rospy.sleep(3)
                self.twist_command(0, 0, 0)
                rospy.sleep(0.5)
                self.reset.example_send_joint_angles(HOME)
                rospy.sleep(6)
                self.reset.example_clear_faults()
                self.Y_pressed = False
            elif self.current_goal_set:
                ur_list = self.compute_ur_for_all_goals()
                goal_idx, confidence = self.infer_goal(ur_list)

                #new
                msg = CentroidConfidenceArray()
                for i, 
                item = CentroidConfidence()
                
                
                blended, alpha = self.blend_inputs(self.uh, ur_list[goal_idx], confidence)
                self.twist_command(*blended)
            else:
                self.twist_command(self.uh.linear_x * 0.5,
                                   self.uh.linear_y * 0.5,
                                   self.uh.linear_z * 0.5)
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node("vosa_teleop")
    node = VOSATeleoperation()
    node.main()
