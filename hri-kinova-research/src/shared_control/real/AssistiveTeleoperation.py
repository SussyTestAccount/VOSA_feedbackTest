#!/usr/bin/env python3

import sys
import os
import math
import utilities
import rospy
import time
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from kortex_driver.msg import Twist
from KinovaRobot import KinovaRobot
from ActuatorModel import ActuatorModel
from sensor_msgs.msg import Joy, JointState
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_driver.msg import BaseCyclic_Feedback, Twist, CartesianSpeed, CartesianReferenceFrame, ActionType, ActionNotification, ConstrainedPose, ActionEvent, Finger, GripperMode, ControllerNotification, ControllerType, Waypoint, AngularWaypoint, WaypointList, TwistCommand
from kortex_driver.srv import Base_ClearFaults, ExecuteAction, ExecuteActionRequest, SetCartesianReferenceFrameRequest, ValidateWaypointList, OnNotificationActionTopic, OnNotificationActionTopicRequest, SendGripperCommand, SendGripperCommandRequest, Stop, ReadActionRequest, ReadAction, SetCartesianReferenceFrame, SendTwistCommand, SendTwistCommandRequest, SendTwistJoystickCommand, SendTwistJoystickCommandRequest


class AssistiveTeleoperation:
    def __init__(self):
        self.robot_name = 'my_gen3'
        self.starting_state = Twist()
        self.uh = Twist()
        
        self.human_input = False
        self.next_user_state = None
        self.next_robot_state = None
        self.current_robot_state = None
        
        self.curr_pos = []
        self.positions = []
        self.goal_predications = []
        self.trajectory_user_inputs = []
        self.ee_position = np.array([0.0, 0.0, 0.0])
        self.ee_pose = np.array([0.0, 0.0, 0.0])

        self.first_goal = np.array([0.58776623, 0.05404127, 0.05432801])  # Rubix Cube
        self.second_goal = np.array([0.58516139, -0.1902532, 0.04022028])  # Tennis Ball Right (Back)
        self.third_goal = np.array([0.3251715,  0.52185464, 0.04072722])  # Tennis Ball Left
        self.fourth_goal = np.array([0.2350669652223587, -0.5779838562011719, 0.24929864704608917])  # Tennis Ball Can

        # Save the angular poses 
        self.first_pose = np.array([146.67155457, -33.41492844,  67.46300507])  # Rubix Cube
        self.second_pose = np.array([128.19418335,   2.62314868,  33.28691483])  # Tennis Ball Right
        self.third_pose = np.array([142.27671814, -12.24828625, 152.54417419])  # Tennis Ball Left
        self.fourth_pose = np.array([164.99588013,  -4.04116392,  12.30979156])  # Tennis Ball Can

        # Rubix Cube Pose
        # [146.67155457 -33.41492844  67.46300507]

        # Tennis Ball Left
        # [142.07815552 -12.24249744 152.88722229]

        # Tennis Ball Right
        # [128.19418335   2.62314868  33.28691483]

        # Tennis Ball Can
        # [164.99588013  -4.04116392  12.30979156]

        # self.list_goals = [self.first_goal, self.second_goal, self.third_goal, self.fourth_goal]
        # self.list_goal_poses = [self.first_pose, self.second_pose, self.third_pose, self.fourth_pose]

        self.list_goals = [np.array([0.38, 0.0, 0.05])]
        self.list_goal_poses = [self.second_pose]
        self.confidences = [0.0 for _ in self.list_goals]
        # Joystick Feedback
        self.subscribe_joy_feedback()
        # Action feedback
        self.jointfeedback = rospy.Subscriber('/' + self.robot_name + "/joint_states", JointState, self.action_feedback_callback, buff_size=1)
        # Base feedback
        self.basefeedback = rospy.Subscriber("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback, self.base_feedback_callback, buff_size=1)
        
        # Services
        send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
        rospy.wait_for_service(send_gripper_command_full_name)
        self.gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

        send_twist_command_full_name = 'my_gen3/base/send_twist_joystick_command'
        rospy.wait_for_service(send_twist_command_full_name)
        self.send_twist_command = rospy.ServiceProxy(send_twist_command_full_name, SendTwistJoystickCommand)
        self.alpha_history = []

        from std_msgs.msg import Float32MultiArray
        from geometry_msgs.msg import Point
        from custom_msgs.msg import CentroidConfidence  # <-- You’ll create this custom message
        
        self.centroid_conf_pub = rospy.Publisher('/goal_confidence_centroids', CentroidConfidenceArray, queue_size=10)


    def plot_goal_confidences(self, ax):
        plt.clf()
        latest_confidences = self.confidences
        ax.bar(range(len(self.list_goals)), latest_confidences, color='blue')  # Create bar graph
        ax.set_xlabel('Goals')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence for Each Goal')
        ax.set_xticks(range(len(self.list_goals)))
        ax.set_xticklabels(['Goal 1', 'Goal 2', 'Goal 3', 'Goal 4'])
        plt.legend()
        plt.draw()
        plt.pause(0.1)  # Pause to update the plot

    def plot_alpha_values(self, ax):
        
        latest_confidences = self.confidences
        plt.cla()
        ax.plot(range(len(self.alpha_history[-100:])), self.alpha_history[-100:], color='blue', label="Robot Control")
        ax.plot(range(len(self.alpha_history[-100:])), 1 - np.array(self.alpha_history[-100:]), color='red', label="Human Control") 
        ax.set_xlabel('Time')
        ax.set_ylabel('Control Weighting')
        ax.set_title('Human vs Robot Control over Time')
        ax.set_ylim(0, 1)
        plt.draw()
        
        plt.pause(0.1)  # Pause to update the plot


    def twist_to_array(self, twist):
        return np.array([twist.linear_x, twist.linear_y, twist.linear_z])

    def subscribe_joy_feedback(self):
        rospy.loginfo("Subscribing to joy node")
        rospy.Subscriber("/joy", Joy, self.controller_callback)

    def twist_command(self, x, y, z, angle_x=0, angle_y=0, angle_z=0):
        twist = Twist()
        twist.linear_x = float(x)
        twist.linear_y = float(y)
        twist.linear_z = float(z)
        
        # Angular Velocities
        # NOTE: In the range [0, 1], in rad/s. 0.5 is fast, keep these numbers low
        twist.angular_x = float(angle_x)
        twist.angular_y = float(angle_y)
        twist.angular_z = float(angle_z)

        twist_command = TwistCommand()
        twist_command.duration = 0
        twist_command.reference_frame = 1
        twist_command.twist = twist

        rospy.loginfo("Sending pose")
        try:
            self.send_twist_command(twist_command)

        except rospy.ServiceException:
            rospy.logerr("Failed to send pose")
            return False

    def controller_callback(self, data):
        controller_position = data.axes
        print(controller_position)
        twist = Twist()
        twist.linear_x = float(controller_position[1])
        twist.linear_y = float(controller_position[0])
        twist.linear_z = float(controller_position[4])
        self.uh = twist

        # Gripper Control
        controller_position = data.axes
        if controller_position[5] != 1.0:
            # self.gripper_value -= 0.1
            self.send_gripper_command(0.0)
        elif controller_position[2] != 1.0:
            # self.gripper_value += 0.1
            self.send_gripper_command(0.65)
        # return True

        if controller_position[1] == 0 and controller_position[0] == 0 and controller_position[4] == 0:
            self.human_input = False

        else:
            self.human_input = True

    def action_feedback_callback(self, feedback):
        return str(feedback.position)

    def send_gripper_command(self, value):
        # Initialize the request
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION
        # Call the service 
        try:
            self.gripper_command(req)
            return True
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True

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

        # Set the end effector position for the class
        self.ee_position[0] = feedback.base.tool_pose_x
        self.ee_position[1] = feedback.base.tool_pose_y
        self.ee_position[2] = feedback.base.tool_pose_z

        # print("EE",self.ee_position[0], self.ee_position[1], self.ee_position[2])
        # Set the end effector pose for the class
        self.ee_pose[0] = feedback.base.tool_pose_theta_x
        self.ee_pose[1] = feedback.base.tool_pose_theta_y
        self.ee_pose[2] = feedback.base.tool_pose_theta_z

    def compute_alpha(self, confidence, method, alpha_min = 0.0, alpha_max = 1.0):
        if method == 'linear':
            alpha = alpha_min + (alpha_max - alpha_min) * confidence
        if method == 'sigmoid':
            k = 10  # steepness of the curve
            alpha = alpha_min + (alpha_max - alpha_min) / (1 + math.exp(-k * confidence - 0.5))
        
        if method == 'piecewise':
            if confidence < 0.4:
                alpha = alpha_min
            elif 0.4 <= confidence < 0.7:
                theta = (alpha_max - alpha_min) / 0.3  # change in alpha / change in confidence = SLOPE
                offset = alpha_min - (theta * 0.4)  # y-intercept
                alpha = theta * confidence + offset  # should give y=mx+c
            else:
                alpha = alpha_max
        alpha = max(min(alpha, alpha_max), alpha_min)  # making sure alpha stays between 0 and 1
        return alpha

    def compute_ur_for_all_goals(self, current_position, current_pose):
        ur_list = list()
        for i, goal in enumerate(self.list_goals):
            direction_vector = goal - current_position
            dir_norm = np.linalg.norm(direction_vector)
            ur = direction_vector / dir_norm
            ###
            if 1 >= dir_norm > 0.02:
                ur = direction_vector
            elif dir_norm <= 0.02:
                ur *= 0
                ur_list.append(np.zeros(6))
                continue
            ###

            desired_pose = self.list_goal_poses[i]
            angular_displacement = desired_pose - current_pose
            angular_norm = np.linalg.norm(angular_displacement)
            angular_velocity = angular_displacement / angular_norm

            ####
            if 1 > angular_norm > 0.4:
                angular_velocity = angular_displacement
            if angular_norm < 0.4:
                angular_velocity *= 0
            ####

            ur_list.append(np.concatenate((ur, angular_velocity)))

        # print("UR List: ", ur_list)
        return ur_list

    def compute_confidence(self, ur, predicted_goal_index, confidence_min=0.0, confidence_max=1.0):
        curr_pos = self.get_curr_pos()

        predicted_goal = self.list_goals[predicted_goal_index]
        dist_to_goal = np.linalg.norm(self.ee_position - predicted_goal)

        w1 = 0.4
        w2 = 0.6

        human_inp = np.array([self.uh.linear_x, self.uh.linear_y, self.uh.linear_z])

        term1 = w1 * np.dot(human_inp, ur[0:3])
        term2 = w2 * math.exp(-dist_to_goal)

        confidence = term1 + term2
        return max(min(confidence, confidence_max), confidence_min)

    def get_robot_command(self, inferred_goal):
        current_position = self.ee_position
        goal_position = self.list_goals[inferred_goal]

        print("Goal Position", goal_position)
        print("Current Position", current_position)

        direction_vector = goal_position - current_position
        ur = direction_vector / np.linalg.norm(direction_vector)

        return ur

    def blend_inputs(self, uh, ur, predicted_goal_index):
        print(ur)
        confidence = self.compute_confidence(ur, predicted_goal_index)
        alpha = self.compute_alpha(confidence, method='piecewise')

        # The human's desired control input
        displacement_h = np.array([uh.linear_x, uh.linear_y, uh.linear_z, 0.0, 0.0, 0.0])  # Note: Human has no control over the angular components of the movement

        # A blending of the two

        # return alpha * displacement_h + (1 - alpha) * ur, alpha
        return alpha * ur + (1 - alpha) * displacement_h, alpha

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

    def calculate_distance_from_goal(self, current_position):
        distances = [np.linalg.norm(goal - current_position) for goal in self.list_goals]
        return distances

    '''
    def infer_goal(self, ur_list):
        # return argmax of confidences
        confidences = [self.compute_confidence(ur_list[i], i) for i in range(len(ur_list))]
        # print("CONFIDENCES: ", confidences)
        best_i = np.argmax(confidences)
        return best_i, confidences[best_i]
    '''
    def infer_goal(self, ur_list):
        raw_confidences = [self.compute_confidence(ur_list[i], i) for i in range(len(ur_list))]
        print("raw-confidences", raw_confidences)
        softmax_confidences = F.softmax(torch.tensor(raw_confidences), dim=0)
        confidences = softmax_confidences.tolist()
        self.confidences = confidences
        inferred_goal_index = torch.argmax(softmax_confidences).item()
        self.confidences = softmax_confidences.tolist()
        print("confidences - ", self.confidences) # 0.24627863806045594, 0.2296496482449283, 0.31338556104623855, 0.21068615264837723
        return inferred_goal_index, confidences[inferred_goal_index]


    def send_gripper_command(self, value):
        # Initialize the request
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        # Call the service 
        try:
            self.gripper_command(req)
            return True
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True

    def cartesian_to_joint_command(self, base, r3_displacement):
        if len(self.curr_pos) == 0:
            print("Not initialized yet, waiting...")
            return
     
        desired_action = Base_pb2.Action()
        desired_action.reach_pose.target_pose.x = self.ee_position[0] + r3_displacement[0]
        desired_action.reach_pose.target_pose.y = self.ee_position[1] + r3_displacement[1]
        desired_action.reach_pose.target_pose.z = self.ee_position[2] + r3_displacement[2]
        desired_action.reach_pose.target_pose.theta_x = self.ee_pose[0]
        desired_action.reach_pose.target_pose.theta_y = self.ee_pose[1]
        desired_action.reach_pose.target_pose.theta_z = self.ee_pose[2]

        return desired_action

    def main(self):

        plt.ion()
        fig, ax = plt.subplots()
        legend_initialized = False # Keep false by default until first loop


        goal_reach_threshold = 0.01
        previous_distances_to_goals = [float('inf')] * len(self.list_goals)
        try:
            robot = KinovaRobot()
            success = robot.is_init_success
            if success:
                success &= robot.clear_faults()
            if success:
                success &= robot.subscribe_base_feedback()
        except Exception as e:
            print(e)
            success = False

        rospy.set_param("is_initialized", success)
        if not success:
            rospy.logerr("The robot initialization encountered an error.")
        else:
            rospy.loginfo("The robot initialization executed without fail.")

        # Create connection to the device and get the router
        with utilities.DeviceConnection.createTcpConnection() as router:
            # Create required services
            base = BaseClient(router)
            base_cyclic = BaseCyclicClient(router)
            teleop = AssistiveTeleoperation()
            rate = rospy.Rate(10)
            try:
                while not rospy.is_shutdown():

                    # self.plot_goal_confidences(ax)
                    self.plot_alpha_values(ax)
                    if not legend_initialized:
                        fig.legend()
                        legend_initialized = True

                    ur_list = self.compute_ur_for_all_goals(self.ee_position, self.ee_pose)
                    inferred_goal, confidence = self.infer_goal(ur_list)

                    msg = CentroidConfidenceArray() #new
                    for i, pos in enumerate(self.list_goals):
                        pt = Point(x=round(pos[0], 2), y=round(pos[1], 2), z=round(pos[2], 2))
                        item = CentroidConfidence()
                        item.centroid = pt
                        item.confidence = float(self.confidences[i])  # Make sure self.confidences is assigned in infer_goal
                        msg.items.append(item)
                    
                    self.centroid_conf_pub.publish(msg)
                   
                    print("ur_list - ", ur_list)

                    self.goal_predications.append(inferred_goal)
                    print(len(self.goal_predications))
                    
                    print("Human in Control? ", self.human_input)

                    # alpha = self.compute_alpha(confidence, method='piecewise')
                    blended_commmand, alpha = self.blend_inputs(self.uh, ur_list[inferred_goal], predicted_goal_index=inferred_goal)
                    self.alpha_history.append(alpha)
                    
                    # Decrease the speed for testing
                    # blended_commmand *= 0.5

                    LINEAR_SPEED = 1.5
                    blended_commmand[0:3] *= LINEAR_SPEED

                    ANGULAR_SPEED = 0.6
                    blended_commmand[3:] *= ANGULAR_SPEED
                    blended_commmand[4] *= -1
                    blended_commmand[5] *= -1
                    
                    # Only move the robot if the human has given input recently
                    # if self.human_input is True:
                    #     self.twist_command(blended_commmand[0], blended_commmand[1], blended_commmand[2], angle_x=blended_commmand[3], angle_y=blended_commmand[4], angle_z=blended_commmand[5])
                    # else:
                    #     self.twist_command(0, 0, 0, angle_x=0, angle_y=0, angle_z=0)
                    self.twist_command(blended_commmand[0], blended_commmand[1], blended_commmand[2], angle_x=blended_commmand[3], angle_y=blended_commmand[4], angle_z=blended_commmand[5])

                    print("Maxed confidence: ", confidence)
                    print("Blended Command = ", blended_commmand)
                    print("inferred_goal: ", inferred_goal, ", Distance to Goal: ", np.linalg.norm(np.array(self.ee_position) - np.array(self.list_goals[inferred_goal])))
                    print("Alpha", alpha)
                    print("UH", self.uh)
                    print("EE: ", self.ee_position)
                    print("EE pose: ", self.ee_pose)

                    for i, goal in enumerate(self.list_goals):
                        current_distance = np.linalg.norm(self.ee_position - goal)
                        previous_distance = previous_distances_to_goals[i]

                        if current_distance < goal_reach_threshold:
                            print("Reached goal ", i+1)
                            self.send_gripper_command(0.5)
                        # elif current_distance > previous_distance:
                        #     print("Moving away from goal")
                        #     self.send_gripper_command(0.0)
                    previous_distances_to_goals[i] = current_distance

                    rate.sleep()

            except rospy.ROSInterruptException:
                print(rospy.ROSInterruptException)
                self.twist_command(0, 0, 0, angle_x=0, angle_y=0, angle_z=0)
            finally:

                # plt.close()
                # plt.ioff()
                # plt.show()
                self.twist_command(0, 0, 0, angle_x=0, angle_y=0, angle_z=0)
    

if __name__ == "__main__":
    rospy.init_node('Assistive_Teleop', anonymous=True)
    asst_teleop = AssistiveTeleoperation()
    asst_teleop.main()

