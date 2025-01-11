#!/usr/bin/env python3

import sys
import time
import rospy

# ROS imports
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Float32, String, Float64, Int32, Bool, Header

from kortex_driver.msg import BaseCyclic_Feedback, Twist, CartesianSpeed, CartesianReferenceFrame, ActionType, \
    ActionNotification, ConstrainedPose, ActionEvent, Finger, GripperMode, ControllerNotification, ControllerType, \
    Waypoint, AngularWaypoint, WaypointList, TwistCommand
from kortex_driver.srv import Base_ClearFaults, ExecuteAction, ExecuteActionRequest, SetCartesianReferenceFrameRequest, \
    ValidateWaypointList, OnNotificationActionTopic, OnNotificationActionTopicRequest, SendGripperCommand, \
    SendGripperCommandRequest, Stop, ReadActionRequest, ReadAction, SetCartesianReferenceFrame, SendTwistCommand, \
    SendTwistCommandRequest, SendTwistJoystickCommand, SendTwistJoystickCommandRequest

from ActuatorModel import ActuatorModel
import threading
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


# from AssistiveTeloperation import AssistiveTeloperation

class KinovaRobot():
    def __init__(self):
        # Initialize commonly used variables
        self.is_init_success = None
        self.last_action_notif_type = None
        self.RETRACT_ACTION_IDENTIFIER = 1
        self.HOME_ACTION_IDENTIFIER = 2
        self.gripper_value = 0.1

        self.robot_name = "my_gen3"
        self.degrees_of_freedom = rospy.get_param("/" + self.robot_name + "/degrees_of_freedom", 7)
        self.is_gripper_present = rospy.get_param("/" + self.robot_name + "/is_gripper_present", True)

        rospy.loginfo("Using robot_name " + self.robot_name + " , robot has " + str(
            self.degrees_of_freedom) + " degrees of freedom and is_gripper_present is " + str(self.is_gripper_present))

        try:
            # Init the services
            self.init_services()

        except rospy.ROSException as e:
            self.is_init_success = False
        else:
            self.is_init_success = True

    def init_services(self):
        time.sleep(0.5)
        rospy.loginfo("Initializing Kinova services")

        clear_faults_full_name = '/' + self.robot_name + '/base/clear_faults'
        rospy.wait_for_service(clear_faults_full_name)
        self.req_clear_faults = rospy.ServiceProxy(clear_faults_full_name, Base_ClearFaults)

        activate_publishing_of_action_notification_full_name = '/' + self.robot_name + '/base/activate_publishing_of_action_topic'
        rospy.wait_for_service(activate_publishing_of_action_notification_full_name)
        self.activate_publishing_of_action_notification = rospy.ServiceProxy(
            activate_publishing_of_action_notification_full_name, OnNotificationActionTopic)

        send_gripper_command_full_name = '/' + self.robot_name + '/base/send_gripper_command'
        rospy.wait_for_service(send_gripper_command_full_name)
        self.gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

        set_cartesian_reference_frame_full_name = '/' + self.robot_name + '/control_config/set_cartesian_reference_frame'
        rospy.wait_for_service(set_cartesian_reference_frame_full_name)
        self.set_cartesian_reference_frame = rospy.ServiceProxy(set_cartesian_reference_frame_full_name,
                                                                SetCartesianReferenceFrame)

        execute_action_full_name = '/' + self.robot_name + '/base/execute_action'
        rospy.wait_for_service(execute_action_full_name)
        self.execute_action = rospy.ServiceProxy(execute_action_full_name, ExecuteAction)

        read_action_full_name = '/' + self.robot_name + '/base/read_action'
        rospy.wait_for_service(read_action_full_name)
        self.read_action = rospy.ServiceProxy(read_action_full_name, ReadAction)

        send_twist_command_full_name = 'my_gen3/base/send_twist_joystick_command'
        rospy.wait_for_service(send_twist_command_full_name)
        self.send_twist_command = rospy.ServiceProxy(send_twist_command_full_name, SendTwistJoystickCommand)

        validate_waypoint_list_full_name = '/' + self.robot_name + '/base/validate_waypoint_list'
        rospy.wait_for_service(validate_waypoint_list_full_name)
        self.validate_waypoint_list = rospy.ServiceProxy(validate_waypoint_list_full_name, ValidateWaypointList)

    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """

        def check(notification, e=e):
            print("EVENT : " + \
                  Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
                    or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()

        return check

    def clear_faults(self):
        try:
            self.req_clear_faults()
        except rospy.ServiceException:
            rospy.logerr("Failed to call ClearFaults")
            return False
        else:
            rospy.loginfo("Cleared the faults successfully")
            return True

    """
    ACTION NOTIFICATION METHODS
    """

    def subscribe_action_notification(self):
        req = OnNotificationActionTopicRequest()
        rospy.loginfo("Activating the action notifications...")
        try:
            self.activate_publishing_of_action_notification(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call OnNotificationActionTopic")
        else:
            rospy.loginfo("Successfully activated the Action Notifications!")

        rospy.loginfo("Subscribing to action topic")
        self.action_topic_sub = rospy.Subscriber("/" + self.robot_name + "/action_topic", ActionNotification,
                                                 callback=self.action_topic_callback)

    def action_topic_callback(self, notif):
        self.last_action_notif_type = notif.action_event

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if (self.last_action_notif_type == ActionEvent.ACTION_END):
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif (self.last_action_notif_type == ActionEvent.ACTION_ABORT):
                rospy.loginfo("Received ACTION_ABORT notification")
                return False
            else:
                time.sleep(0.01)

    """
    BASE FEEDBACK METHODS
    """

    def subscribe_base_feedback(self):
        rospy.loginfo("Subscribing to base feedback")
        rospy.Subscriber("/" + self.robot_name + "/base_feedback", BaseCyclic_Feedback,
                         callback=self.base_feedback_callback, buff_size=10, tcp_nodelay=True)
        return True

    def base_feedback_callback(self, feedback):
        # rospy.loginfo("base feedback callback")
        self.position = ActuatorModel()

        # actuator positions
        self.position.tool_x = feedback.base.tool_pose_x
        self.position.tool_y = feedback.base.tool_pose_y
        self.position.tool_z = feedback.base.tool_pose_z
        self.position.tool_theta_x = feedback.base.tool_pose_theta_x
        self.position.tool_theta_y = feedback.base.tool_pose_theta_y
        self.position.tool_theta_z = feedback.base.tool_pose_theta_z

        gripper_feedback = feedback.interconnect.oneof_tool_feedback.gripper_feedback[0].motor
        self.position.gripper = gripper_feedback[0].position

    """
    CONTROLLER FEEDBACK METHODS
    """

    def subscribe_joy_feedback(self):
        # initialize assistive teleoperation
        # self.assistive_teleoperation = AssistiveTeloperation()

        # subscribed to joystick inputs on topic "joy"
        rospy.loginfo("Subscribing to joy node")
        rospy.Subscriber("/joy", Joy, self.controller_callback)

    def controller_callback(self, data):
        # rospy.loginfo("controller callback")
        controller_position = data.axes

        twist_command = TwistCommand()
        twist_command.duration = 0
        twist_command.reference_frame = 1

        twist = Twist()
        twist.linear_x = float(controller_position[1])
        twist.linear_y = float(controller_position[0])
        twist.linear_z = float(controller_position[4])
        # twist.angular_x = float(controller_position[3])
        # twist.angular_y = float(controller_position[3])
        # twist.angular_z = float(controller_position[5])

        # self.assistive_teleoperation.increment_user_state(twist)

        twist_command.twist = twist

        # rospy.loginfo("Sending pose")

        # Commented this out for testing -- Connor M. 11/09/2023

        # try:
        #     self.send_twist_command(twist_command)

        #     # GRIPPER
        #     if controller_position[5] != 1.0:
        #         #self.gripper_value -= 0.1
        #         self.send_gripper_command(0.0)
        #     elif controller_position[2] != 1.0:
        #        # self.gripper_value += 0.1
        #         self.send_gripper_command(0.65)

        #     # ASSISTIVE TELEOPERATION
        #     #self.assistive_teleoperation.compute_probabilities()

        #     return True
        # except rospy.ServiceException:
        #     rospy.logerr("Failed to send pose")
        #     return False

    """
    ROBOT POSITION COMMANDS
    """

    def send_robot_position(self, identifier_num):
        # The Home Action is used to home the robot. It cannot be deleted and is always ID #2:
        self.last_action_notif_type = None
        req = ReadActionRequest()
        req.input.identifier = identifier_num

        try:
            res = self.read_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ReadAction")
            return False
        # Execute the HOME action if we could read it
        else:
            # What we just read is the input of the ExecuteAction service
            req = ExecuteActionRequest()
            req.input = res.output
            rospy.loginfo("Sending the robot to retracted position...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteAction")
            return False
        else:
            return self.wait_for_action_end_or_abort()

    def send_gripper_command(self, value):
        # Initialize the request
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        # rospy.loginfo("Sending the gripper command...")

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

    def send_data_poses(self, position, pos_num):
        my_cartesian_speed = CartesianSpeed()
        my_cartesian_speed.translation = 0.1  # m/s
        my_cartesian_speed.orientation = 15  # deg/s

        my_constrained_pose = ConstrainedPose()
        my_constrained_pose.constraint.oneof_type.speed.append(my_cartesian_speed)
        # my_constrained_pose.constraint.oneof_type.duration.append(float(10))

        my_constrained_pose.target_pose.x = float(position.tool_x)
        my_constrained_pose.target_pose.y = float(position.tool_y)
        my_constrained_pose.target_pose.z = float(position.tool_z)
        my_constrained_pose.target_pose.theta_x = float(position.tool_theta_x)
        my_constrained_pose.target_pose.theta_y = float(position.tool_theta_y)
        my_constrained_pose.target_pose.theta_z = float(position.tool_theta_z)

        req = ExecuteActionRequest()
        req.input.oneof_action_parameters.reach_pose.append(my_constrained_pose)
        pose_name = "pose " + str(pos_num)
        req.input.name = pose_name
        req.input.handle.action_type = ActionType.REACH_POSE
        req.input.handle.identifier = 1001

        rospy.loginfo("Sending " + pose_name + "...")
        self.last_action_notif_type = None
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send " + pose_name)
            return False
        else:
            rospy.loginfo("Waiting for" + pose_name + "to finish...")

        self.wait_for_action_end_or_abort()
        self.send_gripper_command(float(position.gripper))

        return self.wait_for_action_end_or_abort()

    def cartesian_action_movement(self, x, y, z, base, base_cyclic):
        print("Starting Cartesian action movement ...")
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        feedback = base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = x  # (meters)
        cartesian_pose.y = y - 0.1  # (meters)
        cartesian_pose.z = z - 0.2  # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x  # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y  # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z  # (degrees)

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        print("Executing action")
        base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(20)
        base.Unsubscribe(notification_handle)

        if finished:
            print("Cartesian movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

    def custom_send_joint_angles(self, joint_angles):
        assert len(joint_angles) == 7  # Perhaps change in future

        self.last_action_notif_type = None

        req = ExecuteActionRequest()

        trajectory = WaypointList()
        angular_waypoints = []

        # Each AngularWaypoint needs a duration and the global duration (from WaypointList) is disregarded. 
        # If you put something too small (for either global duration or AngularWaypoint duration), the trajectory will be rejected.
        angular_duration = 0

        angularWaypoint = AngularWaypoint()

        for p in joint_angles:
            angularWaypoint.angles.append(p)

        angularWaypoint.duration = angular_duration

        waypoint = Waypoint()
        waypoint.oneof_type_of_waypoint.angular_waypoint.append(angularWaypoint)
        trajectory.waypoints.append(waypoint)

        angular_waypoints.append(angularWaypoint)

        # Initialize Waypoint and WaypointList
        trajectory.duration = 0
        trajectory.use_optimal_blending = True

        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 30

        while (error_number >= 1 and angular_duration != MAX_ANGULAR_DURATION):
            angular_duration += 1
            for waypoint_indx in range(len(angular_waypoints)):
                trajectory.waypoints[waypoint_indx].oneof_type_of_waypoint.angular_waypoint[
                    0].duration = angular_duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(res.output.trajectory_error_report.trajectory_error_elements)

        if (angular_duration == MAX_ANGULAR_DURATION):
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False

        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)

        # Send the angles
        rospy.loginfo("Sending the robot to recorded data...")
        try:
            print("Execute")
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointjectory")
        else:
            print(self.wait_for_action_end_or_abort())

    def send_joint_angles(self, df_positions):
        self.last_action_notif_type = None

        req = ExecuteActionRequest()

        trajectory = WaypointList()
        angular_waypoints = []

        # Each AngularWaypoint needs a duration and the global duration (from WaypointList) is disregarded. 
        # If you put something too small (for either global duration or AngularWaypoint duration), the trajectory will be rejected.
        angular_duration = 0

        # Angles to send the arm to recorded demonstration data
        for index, position in df_positions.iterrows():
            angularWaypoint = AngularWaypoint()

            angularWaypoint.angles.append(int(position['Position 0']))
            angularWaypoint.angles.append(int(position['Position 1']))
            angularWaypoint.angles.append(int(position['Position 2']))
            angularWaypoint.angles.append(int(position['Position 3']))
            angularWaypoint.angles.append(int(position['Position 4']))
            angularWaypoint.angles.append(int(position['Position 5']))
            angularWaypoint.angles.append(int(position['Position 6']))
            # angularWaypoint.angles.append(int(position['Gripper']))

            angularWaypoint.duration = angular_duration

            waypoint = Waypoint()
            waypoint.oneof_type_of_waypoint.angular_waypoint.append(angularWaypoint)
            trajectory.waypoints.append(waypoint)

            angular_waypoints.append(angularWaypoint)

        # Initialize Waypoint and WaypointList
        trajectory.duration = 0
        trajectory.use_optimal_blending = True

        try:
            res = self.validate_waypoint_list(trajectory)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ValidateWaypointList")
            return False

        error_number = len(res.output.trajectory_error_report.trajectory_error_elements)
        MAX_ANGULAR_DURATION = 30

        while (error_number >= 1 and angular_duration != MAX_ANGULAR_DURATION):
            angular_duration += 1
            for waypoint_indx in range(len(angular_waypoints)):
                trajectory.waypoints[waypoint_indx].oneof_type_of_waypoint.angular_waypoint[
                    0].duration = angular_duration

            try:
                res = self.validate_waypoint_list(trajectory)
            except rospy.ServiceException:
                rospy.logerr("Failed to call ValidateWaypointList")
                return False

            error_number = len(res.output.trajectory_error_report.trajectory_error_elements)

        if (angular_duration == MAX_ANGULAR_DURATION):
            # It should be possible to reach position within 30s
            # WaypointList is invalid (other error than angularWaypoint duration)
            rospy.loginfo("WaypointList is invalid")
            return False

        req.input.oneof_action_parameters.execute_waypoint_list.append(trajectory)

        # Send the angles
        rospy.loginfo("Sending the robot to recorded data...")
        try:
            print("Execute")
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call ExecuteWaypointjectory")
        else:
            print(self.wait_for_action_end_or_abort())

    def set_cartesian_reference_frame(self):
        # Prepare the request with the frame we want to set
        req = SetCartesianReferenceFrameRequest()
        req.input.reference_frame = CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED

        # Call the service
        try:
            self.set_cartesian_reference_frame()
        except rospy.ServiceException:
            rospy.logerr("Failed to call SetCartesianReferenceFrame")
            return False
        else:
            rospy.loginfo("Set the cartesian reference frame successfully")
            return True

        # Wait a bit
        rospy.sleep(0.25)
