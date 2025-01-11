#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy

mode_xy = True


def custom_joy_callback(data):
    global mode_xy
    modified_joy_message = Joy()
    modified_joy_message.header = data.header

    modified_joy_message.axes = list(data.axes)
    modified_joy_message.buttons = list(data.buttons)

    if data.buttons[1] == 1:
        mode_xy = not mode_xy
        rospy.loginfo("Toggled mode to: " + ("x-y" if mode_xy else "z"))

    if mode_xy:
        modified_joy_message.axes[3] = 0
        modified_joy_message.axes[4] = 0
    else:
        modified_joy_message.axes[0] = 0
        modified_joy_message.axes[4] = modified_joy_message.axes[1]

        modified_joy_message.axes[1] = 0
        modified_joy_message.axes[3] = 0



    print(data)
    print(mode_xy)
    pub.publish(modified_joy_message)


if __name__ == '__main__':
    rospy.init_node('custom_joy_node')
    rospy.Subscriber('/joy', Joy, custom_joy_callback)
    pub = rospy.Publisher('/custom_joy_node', Joy, queue_size=10)
    rospy.spin()
