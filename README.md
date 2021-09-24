# turtlebot_handgesture
Python3 script using mediapipe to detect set of hands and send /cmd_vel message (ROS)

PACKAGES

- ROS (tested on kinetic and noetic)

- Python3

- Mediapipe

INSTRUCTIONS

-> Both open = do nothing

-> close right open left = turn right

-> open right close left = turn left

-> close right close left = accelerate forward

-> index finger right, index finger left = accelerate backwards

-> hands out of image = stop 
