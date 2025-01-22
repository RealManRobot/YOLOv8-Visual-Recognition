#!/usr/bin/env python3
# -*- coding=UTF-8 -*-
from std_msgs.msg import String, Bool,Empty
import rospy, sys
from rm_msgs.msg import MoveJ_P,Arm_Current_State,Gripper_Set, Gripper_Pick,ArmState,MoveL,MoveJ,set_modbus_mode,write_register,write_single_register,Tool_Analog_Output
from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation as R
from vi_grab.msg import ObjectInfo
from geometry_msgs.msg import TransformStamped,PointStamped
from geometry_msgs.msg import Point, Quaternion

# 抓取标志位，只抓取一次的时候才用
catch_flag = True

# 相机坐标系物体到机械臂基坐标系转换函数
def convert(x,y,z,x1,y1,z1,rx,ry,rz):
    """
    我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
    机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
    """
    # 相机坐标系到机械臂末端坐标系的旋转矩阵和平移向量
    rotation_matrix = np.array([[ 0.01206237 , 0.99929647  ,0.03551135],
 [-0.99988374 , 0.01172294 , 0.00975125],
 [ 0.00932809 ,-0.03562485 , 0.9993217 ]])
    translation_vector = np.array([-0.08039019, 0.03225555, -0.08256825])
    # 深度相机识别物体返回的坐标
    obj_camera_coordinates = np.array([x, y, z])

    # 机械臂末端的位姿，单位为弧度
    end_effector_pose = np.array([x1, y1, z1,
                                  rx, ry, rz])
    # 将旋转矩阵和平移向量转换为齐次变换矩阵
    T_camera_to_end_effector = np.eye(4)
    T_camera_to_end_effector[:3, :3] = rotation_matrix
    T_camera_to_end_effector[:3, 3] = translation_vector
    # 机械臂末端的位姿转换为齐次变换矩阵
    position = end_effector_pose[:3]
    orientation = R.from_euler('xyz', end_effector_pose[3:], degrees=False).as_matrix()
    T_base_to_end_effector = np.eye(4)
    T_base_to_end_effector[:3, :3] = orientation
    T_base_to_end_effector[:3, 3] = position
    # 计算物体相对于机械臂基座的位姿
    obj_camera_coordinates_homo = np.append(obj_camera_coordinates, [1])  # 将物体坐标转换为齐次坐标
    #obj_end_effector_coordinates_homo = np.linalg.inv(T_camera_to_end_effector).dot(obj_camera_coordinates_homo)
    obj_end_effector_coordinates_homo = T_camera_to_end_effector.dot(obj_camera_coordinates_homo)
    obj_base_coordinates_homo = T_base_to_end_effector.dot(obj_end_effector_coordinates_homo)
    obj_base_coordinates = obj_base_coordinates_homo[:3]  # 从齐次坐标中提取物体的x, y, z坐标
    # 计算物体的旋转
    obj_orientation_matrix = T_base_to_end_effector[:3, :3].dot(rotation_matrix)
    obj_orientation_euler = R.from_matrix(obj_orientation_matrix).as_euler('xyz', degrees=False)
    # 组合结果
    obj_base_pose = np.hstack((obj_base_coordinates, obj_orientation_euler))
    obj_base_pose[3:] = rx,ry,rz
    return obj_base_pose

# 接收到识别物体的回调函数
def object_pose_callback(data):
    # 标志位设置为全局变量
    global catch_flag
    # 判断当前帧的识别结果是否有要抓取的物体
    if data.object_class == object_msg.data and catch_flag :
        print(data)
        # 等待当前的机械臂位姿
        arm_pose_msg = rospy.wait_for_message("/rm_driver/Arm_Current_State", Arm_Current_State, timeout=None)
        print(arm_pose_msg)
        rospy.sleep(1)
        # 等待接收当前机械臂位姿四元数形式
        arm_orientation_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=None)
        print(arm_orientation_msg)
        # 计算机械臂基坐标系下的物体坐标
        result = convert(data.x,data.y,data.z,arm_pose_msg.Pose[0],arm_pose_msg.Pose[1],arm_pose_msg.Pose[2],arm_pose_msg.Pose[3],arm_pose_msg.Pose[4],arm_pose_msg.Pose[5])
        print(data.object_class,':',result)
        # 运动到准备抓取姿态
        before_catch()
        rospy.sleep(3)
        # 抓取物体
        catch(result,arm_orientation_msg)
        print('---------------------------ok------------------------------')
        catch_flag = False
        # 抓取完一次就关闭ros
        rospy.signal_shutdown("****************catch completed****************")

def before_catch():
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=10)
    rospy.sleep(1)
    pic_joint = MoveJ()
    pic_joint.joint = [-0.09342730045318604, -0.8248963952064514, 1.5183943510055542, 0.06789795309305191, 0.8130478262901306, 0.015879500657320023]
    pic_joint.speed = 0.3
    moveJ_pub.publish(pic_joint)
    write_register(1, 43, 2, [0, 1, -122, -96], 1)


def catch(result,arm_orientation_msg):
    moveJ_P_pub = rospy.Publisher("rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=1)
    rospy.sleep(1)
    zero_pose = MoveJ_P()
    zero_pose.Pose.position.x = result[0]+0.05
    zero_pose.Pose.position.y = result[1]
    zero_pose.Pose.position.z = result[2]
    zero_pose.Pose.orientation.x = arm_orientation_msg.Pose.orientation.x
    zero_pose.Pose.orientation.y = arm_orientation_msg.Pose.orientation.y
    zero_pose.Pose.orientation.z =  arm_orientation_msg.Pose.orientation.z
    zero_pose.Pose.orientation.w =  arm_orientation_msg.Pose.orientation.w
    zero_pose.speed = 0.3
    moveJ_P_pub.publish(zero_pose)

    print('*************************catching*************************')
    rospy.sleep(5.0)
    moveL_pub = rospy.Publisher("rm_driver/MoveL_Cmd", MoveL, queue_size=1)
    rospy.sleep(1)
    first_pose = MoveL()
    first_pose.Pose.position.x = result[0]
    first_pose.Pose.position.y = result[1]
    first_pose.Pose.position.z = result[2]
    first_pose.Pose.orientation.x = arm_orientation_msg.Pose.orientation.x
    first_pose.Pose.orientation.y = arm_orientation_msg.Pose.orientation.y
    first_pose.Pose.orientation.z = arm_orientation_msg.Pose.orientation.z
    first_pose.Pose.orientation.w = arm_orientation_msg.Pose.orientation.w
    first_pose.speed = 0.2
    moveL_pub.publish(first_pose)
    rospy.sleep(3)
    write_register(1, 43, 2, [0, 5, 26, -128], 1)  #闭合夹爪

    print('*************************moving*************************')
    rospy.sleep(5.0)

    rospy.sleep(1)
    second_pose = MoveJ_P()
    second_pose.Pose.position.x = -0.118
    second_pose.Pose.position.y = -0.26
    second_pose.Pose.position.z = 0.558
    second_pose.Pose.orientation.x = arm_orientation_msg.Pose.orientation.x
    second_pose.Pose.orientation.y = arm_orientation_msg.Pose.orientation.y
    second_pose.Pose.orientation.z = arm_orientation_msg.Pose.orientation.z
    second_pose.Pose.orientation.w = arm_orientation_msg.Pose.orientation.w
    second_pose.speed = 0.2
    moveJ_P_pub.publish(second_pose)
    rospy.sleep(3)
    print('*************************waiting*************************')

    rospy.sleep(5.0)
    movej_pub = rospy.Publisher("rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    arm_joint_msg = rospy.wait_for_message("/rm_driver/ArmCurrentState", ArmState, timeout=None)
    third_pose = MoveJ()
    third_pose.joint = [arm_joint_msg.joint[0],arm_joint_msg.joint[1],arm_joint_msg.joint[2],arm_joint_msg.joint[3],arm_joint_msg.joint[4],arm_joint_msg.joint[5]-1]
    third_pose.speed = 0.2
    movej_pub.publish(third_pose)
    rospy.sleep(3)
    end_pose = MoveJ()
    end_pose.joint = [arm_joint_msg.joint[0],arm_joint_msg.joint[1],arm_joint_msg.joint[2],arm_joint_msg.joint[3],arm_joint_msg.joint[4],arm_joint_msg.joint[5]]
    end_pose.speed = 0.2
    movej_pub.publish(end_pose)
    rospy.sleep(3)
    print('*************************pouring*************************')

def settask():
    set_pub = rospy.Publisher("rm_driver/Gripper_Set", Gripper_Set, queue_size=1)
    rospy.Duration(1)
    set = Gripper_Set()
    set.position = 1000
    set_pub.publish(set)

def picktask():
    pick_pub = rospy.Publisher("rm_driver/Gripper_Pick_On", Gripper_Pick, queue_size=1)
    rospy.sleep(1)
    pick1 = Gripper_Pick()
    pick1.speed = 200
    pick1.force = 1000
    pick_pub.publish(pick1)

def set_mode():
    pub_modbus_mode = rospy.Publisher("/rm_driver/Set_Modbus_Mode_Cmd", set_modbus_mode, queue_size=1)
    rospy.sleep(1)
    set_modbusmode = set_modbus_mode()
    set_modbus_mode.port = 1
    set_modbus_mode.baudrate = 9600
    set_modbus_mode.timeout = 5
    pub_modbus_mode.publish(set_modbusmode)

def write_register(num1, num2, num3, num4, num5 ):
    pub_write_register = rospy.Publisher("/rm_driver/Write_Register_Cmd", write_register, queue_size=1)
    rospy.sleep(1)
    write_reg = write_register()
    write_reg.port = num1
    write_reg.address = num2
    write_reg.num = num3
    write_reg.data = num4
    write_reg.device = num5
    pub_write_register.publish(write_reg)

def write_single_register():
    pub_write_single_register = rospy.Publisher("/rm_driver/Write_Single_Register_Cmd", write_single_register, queue_size=1)
    rospy.sleep(1)
    write_sin_reg = write_register()
    write_sin_reg.port = 1
    write_sin_reg.address = 45
    write_sin_reg.data = 0
    write_sin_reg.device = 1
    pub_write_single_register.publish(write_sin_reg)

def set_tool():
    pub_tool_voltage = rospy.Publisher("/rm_driver/Tool_Analog_Output",Tool_Analog_Output,queue_size=1)
    rospy.sleep(1)
    set_vol = Tool_Analog_Output()
    Tool_Analog_Output.voltage = 24

    pub_tool_voltage.publish(set_vol)


if __name__ == '__main__':
    rospy.init_node('object_catch')
    set_tool()
    set_mode()
    write_single_register()
    write_register(1, 38, 2, [12, 8, 0, 0], 1)  #打开夹爪
    object_msg = rospy.wait_for_message('/choice_object', String, timeout=None)
    # rospy.Subscriber("xfspeech", String, voice_callback)

    sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
    rospy.spin()
    print('---------------------------final----------------------------')
    



    
