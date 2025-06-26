#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2

class HighestConfidenceViewer:
    def __init__(self):
        rospy.init_node('highest_confidence_viewer', anonymous=True)
        self.bridge = CvBridge()

        # Buffers
        self.image_dict = {}
        self.label_dict = {}
        self.confidence_dict = {}

        self.centroid_queue = []

        # Subscriptions
        rospy.Subscriber('/detected_objects/image', Image, self.image_callback)
        rospy.Subscriber('/detected_objects/label', String, self.label_callback)
        rospy.Subscriber('/detected_objects/centroid', Point, self.centroid_callback)
        rospy.Subscriber('/goal_confidence', Float32MultiArray, self.confidence_callback)

        rospy.loginfo("Initialized viewer using centroid-based matching")

    def centroid_callback(self, msg):
        centroid_key = (round(msg.x, 1), round(msg.y, 1))  # Round for stability
        self.centroid_queue.append(centroid_key)

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.centroid_queue:
                key = self.centroid_queue.pop(0)
                self.image_dict[key] = cv_img
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")

    def label_callback(self, msg):
        if self.centroid_queue:
            key = self.centroid_queue[0]
            self.label_dict[key] = msg.data

    def confidence_callback(self, msg):
        # Assume confidence list is aligned with known centroids in obj_pos
        # We'll map confidence[i] to obj_pos[i]
        # This mapping logic must mirror centroid processing in pick_and_place_vision.py
        obj_pos_list = rospy.get_param("/vision_centroid_keys", [])  # Shared centroid keys
        confidences = msg.data

        for i, conf in enumerate(confidences):
            if i < len(obj_pos_list):
                key = tuple(round(x, 1) for x in obj_pos_list[i][:2])  # Use only x, y
                self.confidence_dict[key] = conf

        self.display_highest_confidence_object()

    def display_highest_confidence_object(self):
        if not self.confidence_dict:
            return

        best_key = max(self.confidence_dict, key=self.confidence_dict.get)
        best_conf = self.confidence_dict[best_key]
        best_img = self.image_dict.get(best_key, None)
        best_label = self.label_dict.get(best_key, "Unknown")

        if best_img is not None:
            title = f"{best_label} (Confidence: {best_conf:.2f})"
            cv2.imshow(title, best_img)
            cv2.waitKey(1)

        # Clear buffers to avoid stale data
        self.image_dict.clear()
        self.label_dict.clear()
        self.confidence_dict.clear()
        self.centroid_queue.clear()

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        viewer = HighestConfidenceViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
