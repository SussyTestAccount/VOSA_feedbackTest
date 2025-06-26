#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
import torch
from std_msgs.msg import String
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from std_msgs.msg import Int32
from ultralytics import YOLO
from geometry_msgs.msg import Point


class ObjectDetection:
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.publisher = rospy.Publisher('/num_objects', Int32, queue_size=10)
        self.image_pub = rospy.Publisher('/detected_objects/image', ROSImage, queue_size=10)
        self.label_pub = rospy.Publisher('/detected_objects/label', String, queue_size=10)
        self.centroid_pub = rospy.Publisher('/detected_objects/centroid', Point, queue_size=10)
        self.bridge = CvBridge()



    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def get_frame(self, data):
        if data.encoding == "rgb8":
            conversion_format = "RGB8"
        elif data.encoding == "bgr8":
            conversion_format = "BGR8"
        else:
            raise ValueError(f"Unsupported encoding {data.encoding}")

        np_arr = np.frombuffer(data.data, dtype=np.uint8)
        image_np = np_arr.reshape(data.height, data.width, -1)

        if conversion_format == "RGB8":
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        self.process_image(image_np)

    def process_image(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cords = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()

        # Filter out boxes with strong overlap
        filtered_boxes = []
        filtered_labels = []
        for i, box in enumerate(cords):
            include = True
            for existing_box in filtered_boxes:
                if self.bb_intersection_over_union(box, existing_box) > 0.90:
                    include = False
                    break
            if include:
                filtered_boxes.append(box)
                filtered_labels.append(labels[i])

        frame, object_count = self.plot_boxes(filtered_labels, filtered_boxes, frame)
        cv2.imshow("Video", frame)
        cv2.waitKey(1)
        self.publisher.publish(object_count)

    def plot_boxes(self, labels, cords, frame):
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        object_count = 0
        for i in range(len(labels)):
            row = cords[i]
            if row[4] >= 0.2:  # Confidence threshold
                object_count += 1
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                label = self.class_to_label(labels[i])

                # Crop the object
                cropped = frame[y1:y2, x1:x2]
                
                # Publish the cropped image and label
                ros_img = self.bridge.cv2_to_imgmsg(cropped, encoding='bgr8')
                self.image_pub.publish(ros_img)
                self.label_pub.publish(String(data=label))
                
                # Still draw the box for visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                # Compute center
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                center_point = Point(x=center_x, y=center_y, z=0.0)
                
                # Publish centroid
                self.centroid_pub.publish(center_point)


        return frame, object_count

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def class_to_label(self, x):
        return self.classes[int(x)]

    def start(self):
        rospy.init_node('object_detection_node', anonymous=True)
        rospy.Subscriber("/camera/color/image_raw", Image, self.get_frame)
        rospy.spin()


if __name__ == '__main__':
    try:
        od = ObjectDetection()
        od.start()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
