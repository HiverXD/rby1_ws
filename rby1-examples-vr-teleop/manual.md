python main.py \
  --local_ip 192.168.0.39 \
  --meta_quest_ip 192.168.0.32 \
  --rby1 192.168.30.1:50051 \
  --rby1_model "a" \
  --no_head
  <!-- --no_gripper -->

python main.py \
  --local_ip 192.168.0.39 \
  --meta_quest_ip 192.168.0.32 \
  --rby1 0.0.0.0:50051 \
  --rby1_model "a" \
  --no_head




# Robot state description 

RobotState(timestamp=datetime.datetime(1970, 1, 1, 0, 12, 25, 86039), is_ready=[ True  True  True  True  True  True  True  True  True  True  True  True
  True  True  True  True  True  True  True  True  True  True  True  True], position=[ 0.448  0.286  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000
  0.000  0.000  0.000 -0.002], velocity=[ 0.000  0.000 -0.000  0.000 -0.000 -0.000  0.000  0.000 -0.000  0.000
 -0.000 -0.000 -0.000  0.000  0.000  0.000 -0.000  0.000  0.000 -0.000
  0.000 -0.000  0.000  0.000], current=[ 0.009 -0.009 -0.640  1.320  1.170  1.362  0.283 -0.090 -0.103 -0.022
 -0.072  0.086 -0.138 -0.163  0.261 -0.088 -0.080  0.039  0.154 -0.244
 -0.096  0.041  0.000  0.005], torque=[ 0.011 -0.011 -9.696 19.998 17.725 13.756  2.858 -0.909 -1.040 -0.222
 -0.727  0.869 -0.759 -0.897  4.369 -0.889 -0.808  0.394  1.555 -1.342
 -0.528  0.686  0.000  0.000], num_collisions=5)
position : <class 'numpy.ndarray'> (24,)


# Serial ID for Gripper 

udevadm info --query=all --name=/dev/ttyUSB0 | grep SHORT

ttyusb0 FT94VRHJ --> master arm 
ttyusb1 FT94VRXZ --> gripper 

sudo gedit /etc/udev/rules.d/99-u2d2.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
ls /dev/rb*

https://rainbowrobotics.github.io/rby1-dev/docs/development/troubleshooting/faq.html#q-the-teleoperation-with-joint-mapping-example-is-not-running-what-could-be-the-cause-and-how-can-i-resolve-it



# 