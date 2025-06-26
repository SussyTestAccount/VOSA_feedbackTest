#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2

class HighestConfidenceViewer:
    def __init__(self):
        rospy.init_node('highest_confidence_viewer', anonymous=True)

        self.bridge = CvBridge()

        # Buffers for current frame
        self.image_buffer = []
        self.label_buffer = []

        self.latest_confidences = []

        # Subscriptions
        rospy.Subscriber('/detected_objects/image', Image, self.image_callback)
        rospy.Subscriber('/detected_objects/label', String, self.label_callback)
        rospy.Subscriber('/goal_confidence', Float32MultiArray, self.confidence_callback)

        rospy.loginfo("Viewer node initialized, waiting for image and confidence data...")

    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_buffer.append(cv_img)
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def label_callback(self, msg):
        self.label_buffer.append(msg.data)

    def confidence_callback(self, msg):
        self.latest_confidences = msg.data
        self.display_highest_confidence_object()

    def display_highest_confidence_object(self):
        if not self.latest_confidences or not self.image_buffer:
            rospy.logwarn("Incomplete data. Waiting for next frame...")
            return

        if len(self.latest_confidences) != len(self.image_buffer):
            rospy.logwarn("Mismatch between number of confidences and images")
            self.clear_buffers()
            return

        # Find the index of the highest confidence
        top_idx = self.latest_confidences.index(max(self.latest_confidences))
        top_image = self.image_buffer[top_idx]
        label = self.label_buffer[top_idx] if top_idx < len(self.label_buffer) else "Unknown"

        # Show the selected image
        title = f"Top Object: {label} (Conf: {self.latest_confidences[top_idx]:.2f})"
        cv2.imshow(title, top_image)
        cv2.waitKey(1)

        self.clear_buffers()

    def clear_buffers(self):
        self.image_buffer = []
        self.label_buffer = []

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        viewer = HighestConfidenceViewer()
        viewer.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
