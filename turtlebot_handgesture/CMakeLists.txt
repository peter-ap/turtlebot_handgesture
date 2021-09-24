cmake_minimum_required(VERSION 3.0.2)
project(turtlebot_handgesture)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
)
catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)




