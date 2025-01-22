#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
import rospy
from std_msgs.msg import String, Bool,Empty

if __name__ == '__main__':
    rospy.init_node('msg_pub',anonymous=True)
    pub_arm_pose = rospy.Publisher("/rm_driver/GetCurrentArmState",Empty,queue_size=1)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub_arm_pose.publish()

        rate.sleep()
