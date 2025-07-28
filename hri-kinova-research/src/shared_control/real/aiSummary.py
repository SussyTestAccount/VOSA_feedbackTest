#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2

from custom_msgs.msg import CentroidConfidence, CentroidConfidenceArray

import openai  # NEEDS PACKAGE DOWNLOADED AND KEY TO WORK!!
import os

openai.api_key = "" #CHANGE THIS LATER TO AN ENV VARIABLE!!

def summarize_intent_from_confidences(centroids, confidences, labels, model="gpt-4", top_n=3):
    ranked = sorted(zip(confidences, centroids, labels), key=lambda x: x[0], reverse=True)[:top_n]
    description_lines = [
        f"{i+1}. Object: {label}, Confidence: {conf:.2f}, Location: ({x:.2f}, {y:.2f})"
        for i, (conf, (x, y), label) in enumerate(ranked)
    ]

    prompt = f"""
Given the following robot confidence values for object goals, produce a one-paragraph summary of what the robot intends to do, based on which object(s) it is most likely trying to reach. Mention object names and relative likelihood.

Object confidence data:
{chr(10).join(description_lines)}
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful robotics summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"].strip()

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
        rospy.Subscriber('/goal_confidence_centroids', CentroidConfidenceArray, self.confidence_callback)

        rospy.loginfo("Initialized viewer using CentroidConfidenceArray messages")

    def centroid_callback(self, msg):
        key = (round(msg.x, 2), round(msg.y, 2))
        self.centroid_queue.append(key)

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
        self.confidence_dict.clear()
        centroid_list = []
        confidence_list = []
        label_list = []
    
        for item in msg.items:
            key = (round(item.centroid.x, 2), round(item.centroid.y, 2))
            self.confidence_dict[key] = item.confidence
    
            # Only include entries we have full info for
            if key in self.centroid_queue and key in self.label_dict:
                centroid_list.append(self.centroid_queue[key])
                confidence_list.append(item.confidence)
                label_list.append(self.label_dict[key])
    
        # Display image
        self.display_highest_confidence_object()
    
        # Generate LLM summary
        if centroid_list and confidence_list and label_list:
            try:
                summary = summarize_intent_from_confidences(centroid_list, confidence_list, label_list)
                rospy.loginfo(f"[Intent Summary] {summary}")
            except Exception as e:
                rospy.logwarn(f"Failed to generate LLM summary: {e}")
    
        # Clean up
        self.image_dict.clear()
        self.label_dict.clear()
        self.confidence_dict.clear()
        self.centroid_queue.clear()

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
