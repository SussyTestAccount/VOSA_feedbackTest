# Toward Zero-Shot User Intent Recognition in Shared Autonomy

### Abstract
A fundamental challenge of shared autonomy is to use high-DoF robots to assist, rather than hinder, humans by first inferring user intent and then empowering the user to achieve their intent. Although successful, prior methods either rely heavily on a priori knowledge of all possible human intents or require many demonstrations and interactions with the human to learn these intents before being able to assist the user. We propose and study a zero-shot, vision-only shared autonomy (VOSA) framework designed to allow robots to use end-effector vision to estimate zero-shot human intents in conjunction with blended control to help humans accomplish manipulation tasks with unknown and dynamically changing object locations. To demonstrate the effectiveness of our VOSA framework, we instantiate a simple version of VOSA on a Kinova Gen3 manipulator and evaluate our system by conducting a user study on three tabletop manipulation tasks. The performance of VOSA matches that of an oracle baseline model that receives privileged knowledge of possible human intents while also requiring significantly less effort than unassisted teleoperation. In more realistic settings, where the set of possible human intents is fully or partially unknown, we demonstrate that VOSA requires less human effort and time than baseline approaches while being preferred by a majority of the participants. Our results demonstrate the efficacy and efficiency of using off-the-shelf vision algorithms to enable flexible and beneficial shared control of a robot manipulator.

[Read the full paper here - [`ArXiv`](https://arxiv.org/pdf/2501.08389)]

---

### Dependencies 

Ensure you have the following dependencies installed:

- [`kinova-ros`](https://github.com/Kinovarobotics/kinova-ros)  
- [`kortex`](https://github.com/Kinovarobotics/ros_kortex)  
- [`realsense-ros`](https://github.com/IntelRealSense/realsense-ros)  

For detailed installation steps, refer to the official documentation of these libraries.

---

### To run the experiments, follow these steps:
1. Add this repository to your ROS workspace:
   ```bash
   cd ~/catkin_ws/src
   git clone https://github.com/aria-lab-code/VOSA.git
   cd ~/catkin_ws
   catkin_make
   ```
   
2. Source your workspace:
   ```bash
   source devel/setup.bash
   ```
3. Launch the central node:
   ```bash
   roslaunch hri-kinova-research shared_control_central.launch
   ```
   
5. Run the following commands for each experiment:
    - **Pick and Place:**  
      ```bash
      roslaunch hri-kinova-research shared_control_pick_n_place.launch
      ```

    - **Deceptive Grasping:**  
      ```bash
      roslaunch hri-kinova-research shared_control_deceptive_grasping.launch
      ```

    - **Shelving:**  
      ```bash
      roslaunch hri-kinova-research shared_control_shelving.launch
      ```

  ### Cite this work (BibTeX)
  ```
@inproceedings{10.5555/3721488.3721519,
author = {Belsare, Atharv and Karimi, Zohre and Mattson, Connor and Brown, Daniel S.},
title = {Toward Zero-Shot User Intent Recognition in Shared Autonomy},
year = {2025},
publisher = {IEEE Press},
abstract = {A fundamental challenge of shared autonomy is to use high-DoF robots to assist, rather than hinder, humans by first inferring user intent and then empowering the user to achieve their intent. Although successful, prior methods either rely heavily on a priori knowledge of all possible human intents or require many demonstrations and interactions with the human to learn these intents before being able to assist the user. We propose and study a zero-shot, vision-only shared autonomy (VOSA) framework designed to allow robots to use end-effector vision to estimate zero-shot human intents in conjunction with blended control to help humans accomplish manipulation tasks with unknown and dynamically changing object locations. To demonstrate the effectiveness of our VOSA framework, we instantiate a simple version of VOSA on a Kinova Gen3 manipulator and evaluate our system by conducting a user study on three tabletop manipulation tasks. The performance of VOSA matches that of an oracle baseline model that receives privileged knowledge of possible human intents while also requiring significantly less effort than unassisted teleoperation. In more realistic settings, where the set of possible human intents is fully or partially unknown, we demonstrate that VOSA requires less human effort and time than baseline approaches while being preferred by a majority of the participants. Our results demonstrate the efficacy and efficiency of using off-the-shelf vision algorithms to enable flexible and beneficial shared control of a robot manipulator. Code and videos available here: https://sites.google.com/view/zeroshot-sharedautonomy/home},
booktitle = {Proceedings of the 2025 ACM/IEEE International Conference on Human-Robot Interaction},
pages = {222–231},
numpages = {10},
keywords = {computer vision, shared autonomy, teleoperation},
location = {Melbourne, Australia},
series = {HRI '25}
}
  ```
