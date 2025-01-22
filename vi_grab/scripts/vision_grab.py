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

# 相机坐标系到机械臂末端坐标系的旋转矩阵，通过手眼标定得到
rotation_matrix = np.array([[0.01206237, 0.99929647, 0.03551135],
                            [-0.99988374, 0.01172294, 0.00975125],
                            [0.00932809, -0.03562485, 0.9993217]])
# 相机坐标系到机械臂末端坐标系的平移向量，通过手眼标定得到
translation_vector = np.array([-0.08039019, 0.03225555, -0.08256825])

# 相机坐标系物体到机械臂基坐标系转换函数
def convert(x,y,z,x1,y1,z1,rx,ry,rz):
    """
    函数功能：我们需要将旋转向量和平移向量转换为齐次变换矩阵，然后使用深度相机识别到的物体坐标（x, y, z）和
    机械臂末端的位姿（x1,y1,z1,rx,ry,rz）来计算物体相对于机械臂基座的位姿（x, y, z, rx, ry, rz）
    输入参数：深度相机识别到的物体坐标（x, y, z）和机械臂末端的位姿（x1,y1,z1,rx,ry,rz）
    返回值：物体在机械臂基座坐标系下的位置（x, y, z）
    """
    global rotation_matrix,translation_vector
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
    """
    函数功能：每帧图像经过识别后的回调函数，若有抓取指令，则判断当前画面帧中是否有被抓物体，如果有则将物体坐标进行转换，并让机械臂执行抓取动作
    输入参数：无
    返回值：无
    """

    # 判断当前帧的识别结果是否有要抓取的物体
    if data.object_class == object_msg.data and len(object_msg.data) > 0 :

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
        # 抓取物体
        catch(result,arm_orientation_msg)
        # 清除object_msg的信息，之后二次发布抓取物体信息可以再执行
        object_msg.data = ''


def movej_type(joint,speed):
    '''
    函数功能：通过输入机械臂每个关节的数值（弧度），让机械臂以指定速度（0-1，最好小于0.5，否则太快）运动到指定姿态
    输入参数：[joint1,joint2,joint3,joint4,joint5,joint6]、speed
    返回值：无
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    move_joint = MoveJ()
    move_joint.joint = joint
    move_joint.speed = speed
    moveJ_pub.publish(move_joint)


def movejp_type(pose,speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度（0-1，最好小于0.5，否则太快）运动到指定位姿
    输入参数：pose（position.x、position.y、position.z、orientation.x、orientation.y、orientation.z、orientation.w）、speed
    返回值：无
    '''
    moveJ_P_pub = rospy.Publisher("rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=1)
    rospy.sleep(1)
    move_joint_pose = MoveJ_P()
    move_joint_pose.Pose.position.x = pose[0]
    move_joint_pose.Pose.position.y = pose[1]
    move_joint_pose.Pose.position.z = pose[2]
    move_joint_pose.Pose.orientation.x = pose[3]
    move_joint_pose.Pose.orientation.y = pose[4]
    move_joint_pose.Pose.orientation.z =  pose[5]
    move_joint_pose.Pose.orientation.w =  pose[6]
    move_joint_pose.speed = speed
    moveJ_P_pub.publish(move_joint_pose)


def movel_type(pose,speed):
    '''
    函数功能：通过输入机械臂末端的位姿数值，让机械臂以指定速度（0-1，最好小于0.5，否则太快）直线运动到指定位姿
    输入参数：pose（position.x、position.y、position.z、orientation.x、orientation.y、orientation.z、orientation.w）、speed
    返回值：无
    '''
    moveL_pub = rospy.Publisher("rm_driver/MoveL_Cmd", MoveL, queue_size=1)
    rospy.sleep(1)
    move_line_pose = MoveL()
    move_line_pose.Pose.position.x = pose[0]
    move_line_pose.Pose.position.y = pose[1]
    move_line_pose.Pose.position.z = pose[2]
    move_line_pose.Pose.orientation.x = pose[3]
    move_line_pose.Pose.orientation.y = pose[4]
    move_line_pose.Pose.orientation.z =  pose[5]
    move_line_pose.Pose.orientation.w =  pose[6]
    move_line_pose.speed = speed
    moveL_pub.publish(move_line_pose)

def arm_ready_pose():
    '''
    函数功能：执行整个抓取流程前先运动到一个能够稳定获取物体坐标信息的姿态，让机械臂在此姿态下获取识别物体的三维坐标，机械臂以关节运动的方式到达拍照姿态，
    此关节数值可以根据示教得到，将机械臂通过按住绿色按钮拖动到能够获取较好效果的姿态
    输入参数：无
    返回值：无
    '''
    moveJ_pub = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=1)
    rospy.sleep(1)
    pic_joint = MoveJ()
    pic_joint.joint = [-0.09342730045318604, -0.8248963952064514, 1.5183943510055542, 0.06789795309305191, 0.8130478262901306, 0.015879500657320023]
    pic_joint.speed = 0.3
    moveJ_pub.publish(pic_joint)


def catch(result,arm_orientation_msg):
    '''
    函数功能：机械臂执行抓取动作
    输入参数：经过convert函数转换得到的‘result’和机械臂当前的四元数位姿‘arm_orientation_msg’
    返回值：无
    '''
    # 上一步通过pic_joint运动到了识别较好的姿态，然后就开始抓取流程
    # 流程第一步：经过convert转换后，得到了机械臂坐标系下的物体位置坐标result，通过movej_p运动到result目标附近，因为不能一下就到达
    movejp_type([result[0]+0.07,result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************catching  step1*************************')

    # 抓取第二步：通过抓取第一步已经到达了物体前方，后续使用movel运动方式让机械臂直线运动到物体坐标处
    movel_type([result[0],result[1],result[2],arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************catching  step2*************************')

    # 抓取第三步：到达目标处，闭合夹爪
    gripper_close()
    print('*************************catching  step3*************************')

    # 倒水第一步：抬起物体5厘米
    movel_type([result[0],result[1],result[2]+0.05,arm_orientation_msg.Pose.orientation.x,arm_orientation_msg.Pose.orientation.y,
                 arm_orientation_msg.Pose.orientation.z,arm_orientation_msg.Pose.orientation.w],0.3)
    print('*************************pour  step1*************************')

    # 倒水第二步：旋转机械臂末端关节,通过给机械臂末端关节添加转动弧度达到倒水的动作示例
    movej_type([arm_orientation_msg.joint[0],arm_orientation_msg.joint[1],arm_orientation_msg.joint[2],arm_orientation_msg.joint[3],
                arm_orientation_msg.joint[4],arm_orientation_msg.joint[5]+1],0.3)
    print('*************************pour  step2*************************')


def set_mode():
    '''
    函数功能：设置modbus模式
    输入参数：无
    返回值：无
    '''
    pub_modbus_mode = rospy.Publisher("/rm_driver/Set_Modbus_Mode_Cmd", set_modbus_mode, queue_size=1)
    rospy.sleep(1)
    set_modbus_mode = set_modbus_mode()
    set_modbus_mode.port = 1
    set_modbus_mode.baudrate = 9600
    set_modbus_mode.timeout = 5
    pub_modbus_mode.publish(set_modbusmode)

def modbus_gripper_set(num1, num2, num3, num4, num5 ):
    '''
    函数功能：写多个寄存器
    输入参数：port、address、num、data、device
    返回值：无
    '''
    pub_write_register = rospy.Publisher("/rm_driver/Write_Register_Cmd", write_register, queue_size=1)
    rospy.sleep(1)
    write_reg = write_register()
    write_reg.port = num1
    write_reg.address = num2
    write_reg.num = num3
    write_reg.data = num4
    write_reg.device = num5
    pub_write_register.publish(write_reg)

def modbus_gripper_control():
    '''
    函数功能：写单个寄存器
    输入参数：无
    返回值：无
    '''
    pub_write_single_register = rospy.Publisher("/rm_driver/Write_Single_Register_Cmd", write_single_register, queue_size=1)
    rospy.sleep(1)
    write_sin_reg = write_register()
    write_sin_reg.port = 1
    write_sin_reg.address = 45
    write_sin_reg.data = 0
    write_sin_reg.device = 1
    pub_write_single_register.publish(write_sin_reg)

def set_tool():
    '''
    函数功能：设置工具端电压输出
    输入参数：无
    返回值：无
    '''
    pub_tool_voltage = rospy.Publisher("/rm_driver/Tool_Analog_Output",Tool_Analog_Output,queue_size=1)
    rospy.sleep(1)
    set_vol = Tool_Analog_Output()
    set_vol.voltage = 24
    pub_tool_voltage.publish(set_vol)


def gripper_open():
    '''
    函数功能：打开4C2夹爪
    输入参数：无
    返回值：无
    '''
    set_pub = rospy.Publisher("rm_driver/Gripper_Set", Gripper_Set, queue_size=1)
    rospy.sleep(1)
    set = Gripper_Set()
    set.position = 1000
    set_pub.publish(set)

def gripper_close():
    '''
    函数功能：闭合4C2夹爪
    输入参数：无
    返回值：无
    '''
    pick_pub = rospy.Publisher("rm_driver/Gripper_Pick_On", Gripper_Pick, queue_size=1)
    rospy.sleep(1)
    pick1 = Gripper_Pick()
    pick1.speed = 200
    pick1.force = 1000
    pick_pub.publish(pick1)

if __name__ == '__main__':
    rospy.init_node('object_catch')
    pub_arm_pose = rospy.Publisher("/rm_driver/GetCurrentArmState",Empty,queue_size=1)
    gripper_open()   #初始化打开夹爪
    object_msg = rospy.wait_for_message('/choice_object', String, timeout=None)
    sub_object_pose = rospy.Subscriber("/object_pose", ObjectInfo, object_pose_callback, queue_size=1)
    rospy.spin()