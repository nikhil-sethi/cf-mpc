#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/

# MIT License

# Copyright (c) 2022 Bitcraze

# @file crazyflie_controllers_py.py
# Controls the crazyflie motors in webots in Python

"""crazyflie_controller_py controller."""


from controller import Robot
from controller import Supervisor
from controller import Motor
from controller import InertialUnit
from controller import GPS
from controller import Gyro
from controller import Keyboard
from controller import Camera
from controller import DistanceSensor
import numpy as np
from math import cos, sin, degrees, radians
from model import MPCControllerAcados,Crazyflie

import sys
# Change this path to your crazyflie-firmware folder
sys.path.append('/home/nikhil/Software/crazyflie/crazyflie-firmware')
import cffirmware

robot = Robot()
super = Supervisor()
timestep = int(robot.getBasicTimeStep())

## Initialize motors
m1_motor = robot.getDevice("m1_motor")
m1_motor.setPosition(float('inf'))
m1_motor.setVelocity(-1)
m2_motor = robot.getDevice("m2_motor")
m2_motor.setPosition(float('inf'))
m2_motor.setVelocity(1)
m3_motor = robot.getDevice("m3_motor")
m3_motor.setPosition(float('inf'))
m3_motor.setVelocity(-1)
m4_motor = robot.getDevice("m4_motor")
m4_motor.setPosition(float('inf'))
m4_motor.setVelocity(1)

## Initialize Sensors
imu = robot.getDevice("inertial_unit")
imu.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)
gyro = robot.getDevice("gyro")
gyro.enable(timestep)
camera = robot.getDevice("camera")
camera.enable(timestep)
range_front = robot.getDevice("range_front")
range_front.enable(timestep)
range_left = robot.getDevice("range_left")
range_left.enable(timestep)
range_back = robot.getDevice("range_back")
range_back.enable(timestep)
range_right = robot.getDevice("range_right")
range_right.enable(timestep)

## Get keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

## Initialize variables
pastXGlobal = 0
pastYGlobal = 0
pastZGlobal = 0

past_time = robot.getTime()

cffirmware.controllerPidInit()

print('Take off!')

controller = MPCControllerAcados(agent=Crazyflie())

trail_line_set_node = super.getFromDef("POINT_SET")
coord_node = trail_line_set_node.getField("coord").getSFNode()
coords = coord_node.getField("point")
coord_idx = trail_line_set_node.getField("coordIndex")
print(coords)
pts = np.zeros(2*3)

coords.setMFVec3f(0, [1,1,1])

# Main loop:
while robot.step(timestep) != -1:

    dt = robot.getTime() - past_time

    ## Get measurements
    roll = imu.getRollPitchYaw()[0]
    pitch = imu.getRollPitchYaw()[1]
    yaw = imu.getRollPitchYaw()[2]
    quat = imu.getQuaternion()
    roll_rate = gyro.getValues()[0]
    pitch_rate = gyro.getValues()[1]
    yaw_rate = gyro.getValues()[2]
    xGlobal = gps.getValues()[0]
    vxGlobal = (xGlobal - pastXGlobal)/dt
    yGlobal = gps.getValues()[1]
    vyGlobal = (yGlobal - pastYGlobal)/dt
    zGlobal = gps.getValues()[2]
    vzGlobal = (zGlobal - pastZGlobal)/dt


    ## Put measurement in state estimate
    # TODO replace these with a EKF python binding
    state = cffirmware.state_t()
    state.attitude.roll = degrees(roll)
    state.attitude.pitch = -degrees(pitch)
    state.attitude.yaw = degrees(yaw)
    state.position.x = xGlobal
    state.position.y = yGlobal
    state.position.z = zGlobal
    state.velocity.x = vxGlobal
    state.velocity.y = vyGlobal
    state.velocity.z = vzGlobal

    # Put gyro in sensor data
    sensors = cffirmware.sensorData_t()
    sensors.gyro.x = degrees(roll_rate)
    sensors.gyro.y = degrees(pitch_rate)
    sensors.gyro.z = degrees(yaw_rate)

    # keyboard input
    forwardDesired = 0
    sidewaysDesired = 0
    yawDesired = 0

    key = keyboard.getKey()
    while key>0:
        if key == Keyboard.UP:
            forwardDesired = 0.5
        elif key == Keyboard.DOWN:
            forwardDesired = -0.5
        elif key == Keyboard.RIGHT:
            sidewaysDesired = -0.5
        elif key == Keyboard.LEFT:
            sidewaysDesired = 0.5
        elif key == ord('Q'):
            yawDesired = 8
        elif key == ord('E'):
            yawDesired = -8

        key = keyboard.getKey()

    ## Example how to get sensor data
    # range_front_value = range_front.getValue();
    # cameraData = camera.getImage()

    ## Fill in Setpoints
    setpoint = cffirmware.setpoint_t()
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.position.z = 1.0
    setpoint.mode.yaw = cffirmware.modeVelocity
    setpoint.attitudeRate.yaw = degrees(yawDesired)
    setpoint.mode.x = cffirmware.modeVelocity
    setpoint.mode.y = cffirmware.modeVelocity
    setpoint.velocity.x = forwardDesired
    setpoint.velocity.y = sidewaysDesired
    setpoint.velocity_body = True

    ## Firmware PID bindings
    control = cffirmware.control_t()
    tick = 100 #this value makes sure that the position controller and attitude controller are always always initiated
    cffirmware.controllerPid(control, setpoint,sensors,state,tick)


    # sim_state = np.array([xGlobal, yGlobal, zGlobal, quat[3], quat[0], quat[1], quat[2], vxGlobal, vyGlobal, vzGlobal, roll_rate, pitch_rate, yaw_rate])
    
    sim_state = np.array([xGlobal, yGlobal, zGlobal, roll, pitch, yaw, vxGlobal, vyGlobal, vzGlobal, roll_rate, pitch_rate, yaw_rate])


    action = controller.get_action(state_d=setpoint, state_c=sim_state)
    
    for i in range(50):
        preds_i = controller.solver.get(i, "x")
        # print(preds_i[:3])
        coords.setMFVec3f(i, preds_i[:3])

    # print("sim state:", sim_state)
    # print("mpc state:", controller.x0[2])
    print("mpc action:", controller.u0)

    # print("state diff:", np.abs(sim_state)-np.abs(controller.x0))
    ##
    cmd_roll = radians(control.roll)
    cmd_pitch = radians(control.pitch)
    cmd_yaw = -radians(control.yaw)
    cmd_thrust = control.thrust

    ## Motor mixing
    motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
    motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
    motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
    motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

    
    scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
    
    ### pid controller
    print("pid action:",[-motorPower_m1/scaling, motorPower_m2/scaling, -motorPower_m3/scaling, motorPower_m4/scaling])
    # m1_motor.setVelocity(-motorPower_m1/scaling)
    # m2_motor.setVelocity(motorPower_m2/scaling)
    # m3_motor.setVelocity(-motorPower_m3/scaling)
    # m4_motor.setVelocity(motorPower_m4/scaling)

    ### mpc actions
    # m1_motor.setVelocity(-action[0])
    # m2_motor.setVelocity(action[1])
    # m3_motor.setVelocity(-action[2])
    # m4_motor.setVelocity(action[3])


    test_action = [30,30,30,30]
    scale = 50
    m1_motor.setVelocity(-action[0]*scale)
    m2_motor.setVelocity(action[1]*scale)
    m3_motor.setVelocity(-action[2]*scale)
    m4_motor.setVelocity(action[3]*scale)

    past_time = robot.getTime()
    pastXGlobal = xGlobal
    pastYGlobal = yGlobal
    pastZGlobal = zGlobal

    pass
